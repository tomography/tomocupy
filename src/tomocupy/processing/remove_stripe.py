#!/usr/bin/env python
# -*- coding: utf-8 -*-

# *************************************************************************** #
#                  Copyright © 2022, UChicago Argonne, LLC                    #
#                           All Rights Reserved                               #
#                         Software Name: Tomocupy                             #
#                     By: Argonne National Laboratory                         #
#                                                                             #
#                           OPEN SOURCE LICENSE                               #
#                                                                             #
# Redistribution and use in source and binary forms, with or without          #
# modification, are permitted provided that the following conditions are met: #
#                                                                             #
# 1. Redistributions of source code must retain the above copyright notice,   #
#    this list of conditions and the following disclaimer.                    #
# 2. Redistributions in binary form must reproduce the above copyright        #
#    notice, this list of conditions and the following disclaimer in the      #
#    documentation and/or other materials provided with the distribution.     #
# 3. Neither the name of the copyright holder nor the names of its            #
#    contributors may be used to endorse or promote products derived          #
#    from this software without specific prior written permission.            #
#                                                                             #
#                                                                             #
# *************************************************************************** #
#                               DISCLAIMER                                    #
#                                                                             #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS         #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT           #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS           #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT    #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,      #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED    #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR      #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF      #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING        #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS          #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                #
# *************************************************************************** #

import cupy as cp
import pywt
from functools import lru_cache
from cupyx.scipy.ndimage import median_filter
from cupyx.scipy.ndimage import binary_dilation
from cupyx.scipy.ndimage import uniform_filter1d

__all__ = ['DWTForward', 'DWTInverse', 'afb1d', 'remove_all_stripe', 'remove_stripe_fw', 'remove_stripe_ti']

###### Ring removal with wavelet filtering (adapted for cupy from pytroch_wavelet package https://pytorch-wavelets.readthedocs.io/)##########

# Both filter banks run their whole tap loop inside a single kernel: one
# thread owns one output sample and walks the taps in registers, so the array
# is read once instead of once per tap. Written as a Python loop over taps,
# a 32-tap wavelet (the sym16 default) costs 32 strided passes over the full
# array in each direction, which is what used to dominate remove_stripe_fw.
# The symmetric padding is folded in too, so the full-size gather copy it
# needed is gone as well.

# pywt 'symmetric': half-sample reflection of t into [0, N). The modulo form
# keeps working when the pad is wider than the signal, which does happen at
# deep levels with long filters.
_REFLECT = r'''
    if (t < 0 || t >= N) {
        int p = 2 * N;
        int m = t % p;
        if (m < 0) m += p;
        t = (m < N) ? m : (p - 1 - m);
    }
'''

_afb_row = cp.ElementwiseKernel(
    'raw float32 x, raw float32 h0, raw float32 h1, '
    'int32 L, int32 N, int32 OS, int32 M1, int32 H',
    'raw float32 out',
    '''
    int o   = i % OS;
    int bch = i / OS;                   // (b*C + c)*H + h
    int h   = bch % H;
    long long base = (long long)bch * N;

    int t0 = 2 * o - M1;
    float a = 0.f, b = 0.f;
    if (t0 >= 0 && t0 + L <= N) {       // interior: no reflection needed
        for (int j = 0; j < L; ++j) {
            float v = x[base + t0 + j];
            a += h0[j] * v;  b += h1[j] * v;
        }
    } else {
        for (int j = 0; j < L; ++j) {
            int t = t0 + j;
''' + _REFLECT + '''
            float v = x[base + t];
            a += h0[j] * v;  b += h1[j] * v;
        }
    }
    long long ob = ((long long)(bch / H) * 2 * H + h) * OS + o;
    out[ob]                     = a;    // lowpass
    out[ob + (long long)H * OS] = b;    // highpass
    ''', 'afb_row')

_afb_col = cp.ElementwiseKernel(
    'raw float32 x, raw float32 h0, raw float32 h1, '
    'int32 L, int32 N, int32 OS, int32 M1, int32 W',
    'raw float32 out',
    '''
    int w  = i % W;                     // fastest axis, so loads coalesce
    int r  = i / W;
    int o  = r % OS;
    long long bc = r / OS;              // b*C + c
    long long base = bc * (long long)N * W + w;

    int t0 = 2 * o - M1;
    float a = 0.f, b = 0.f;
    if (t0 >= 0 && t0 + L <= N) {
        for (int j = 0; j < L; ++j) {
            float v = x[base + (long long)(t0 + j) * W];
            a += h0[j] * v;  b += h1[j] * v;
        }
    } else {
        for (int j = 0; j < L; ++j) {
            int t = t0 + j;
''' + _REFLECT + '''
            float v = x[base + (long long)t * W];
            a += h0[j] * v;  b += h1[j] * v;
        }
    }
    long long ob = (bc * 2 * OS + o) * W + w;
    out[ob]                     = a;
    out[ob + (long long)OS * W] = b;
    ''', 'afb_col')


def afb1d(x, h0, h1='zero', dim=-1):
    """1D analysis filter bank: stride-2 convolution along dim, all channels in parallel.

    Output channels interleaved [lo_0, hi_0, lo_1, hi_1, ...] to match
    the grouped-convolution ordering expected by DWTForward.
    """
    C = x.shape[1]
    d = dim % 4
    N = x.shape[d]
    h0f = cp.ascontiguousarray(h0.ravel(), dtype='float32')
    h1f = cp.ascontiguousarray(h1.ravel(), dtype='float32')
    L = int(h0f.size)
    outsize = pywt.dwt_coeff_len(N, L, mode='symmetric')
    # left pad width; the right pad only has to exist, the kernel never reads
    # past the taps it needs
    m1 = (2 * (outsize - 1) - N + L) // 2
    x = cp.ascontiguousarray(x, dtype='float32')
    B = x.shape[0]
    if d == 3:  # row direction: stride-2 along axis 3
        H = x.shape[2]
        out = cp.empty((B, C, 2, H, outsize), dtype='float32')
        _afb_row(x, h0f, h1f, L, N, outsize, m1, H, out,
                 size=B * C * H * outsize)
    else:  # col direction: stride-2 along axis 2
        W = x.shape[3]
        out = cp.empty((B, C, 2, outsize, W), dtype='float32')
        _afb_col(x, h0f, h1f, L, N, outsize, m1, W, out,
                 size=B * C * outsize * W)
    return out.reshape(B, 2*C, *out.shape[3:])


# The synthesis bank is the same idea turned inside out. Upsample-and-filter
# is naturally a scatter, out[j : j+2W : 2] += g[j]*lo + g[j]*hi for every
# tap, which means L read-modify-write passes over the output. Inverting it
# into a gather writes each output exactly once:
#   out[q] = sum_{j = q (mod 2)} g0[j]*lo[(q-j)/2] + g1[j]*hi[(q-j)/2]
# and only q in the range that survives the trim, q = qout + (L-2), is
# evaluated at all.
_sfb_row = cp.ElementwiseKernel(
    'raw float32 lo, raw float32 hi, raw float32 g0, raw float32 g1, '
    'int32 L, int32 W, int32 OW',
    'float32 out',
    '''
    int q = i % OW + L - 2;
    long long base = (long long)(i / OW) * W;
    float a = 0.f;
    for (int j = q & 1; j < L; j += 2) {
        int w = (q - j) >> 1;
        if (w >= 0 && w < W) a += g0[j] * lo[base + w] + g1[j] * hi[base + w];
    }
    out = a;
    ''', 'sfb_row')

_sfb_col = cp.ElementwiseKernel(
    'raw float32 lo, raw float32 hi, raw float32 g0, raw float32 g1, '
    'int32 L, int32 H, int32 OH, int32 W',
    'float32 out',
    '''
    int w = i % W;
    int r = i / W;
    int q = r % OH + L - 2;
    long long base = (long long)(r / OH) * H * W + w;
    float a = 0.f;
    for (int j = q & 1; j < L; j += 2) {
        int h = (q - j) >> 1;
        if (h >= 0 && h < H)
            a += g0[j] * lo[base + (long long)h * W]
               + g1[j] * hi[base + (long long)h * W];
    }
    out = a;
    ''', 'sfb_col')


def sfb1d(lo, hi, g0, g1='zero', dim=-1):
    """1D synthesis filter bank: upsample by 2 along dim and filter."""
    C = lo.shape[1]
    d = dim % 4
    g0f = cp.ascontiguousarray(g0.ravel(), dtype='float32')
    g1f = cp.ascontiguousarray(g1.ravel(), dtype='float32')
    L = int(g0f.size)
    B = lo.shape[0]
    lo = cp.ascontiguousarray(lo, dtype='float32')
    hi = cp.ascontiguousarray(hi, dtype='float32')
    H, W = lo.shape[2], lo.shape[3]
    if d == 3:  # row direction: stride (1, 2)
        ow = (W - 1) * 2 + L - 2 * (L - 2)
        out = cp.empty((B, C, H, ow), dtype='float32')
        _sfb_row(lo, hi, g0f, g1f, L, W, ow, out)
    else:  # col direction: stride (2, 1)
        oh = (H - 1) * 2 + L - 2 * (L - 2)
        out = cp.empty((B, C, oh, W), dtype='float32')
        _sfb_col(lo, hi, g0f, g1f, L, H, oh, W, out)
    return out

class DWTForward():
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        wave (str): Which wavelet to use.                    
        """

    def __init__(self, wave='db1'):
        super().__init__()

        wave = pywt.Wavelet(wave)
        h0_col, h1_col = wave.dec_lo, wave.dec_hi
        h0_row, h1_row = h0_col, h1_col

        self.h0_col = cp.array(h0_col).astype('float32')[
            ::-1].reshape((1, 1, -1, 1))
        self.h1_col = cp.array(h1_col).astype('float32')[
            ::-1].reshape((1, 1, -1, 1))
        self.h0_row = cp.array(h0_row).astype('float32')[
            ::-1].reshape((1, 1, 1, -1))
        self.h1_row = cp.array(h1_row).astype('float32')[
            ::-1].reshape((1, 1, 1, -1))

    def apply(self, x):
        """ Forward pass of the DWT.

        Args:
            x (array): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        # Do a multilevel transform
        # Do 1 level of the transform
        lohi = afb1d(x, self.h0_row, self.h1_row, dim=3)
        y = afb1d(lohi, self.h0_col, self.h1_col, dim=2)
        s = y.shape
        y = y.reshape(s[0], -1, 4, s[-2], s[-1])
        x = cp.ascontiguousarray(y[:, :, 0])
        yh = cp.ascontiguousarray(y[:, :, 1:])
        return x, yh

class DWTInverse():
    """ Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str): Which wavelet to use.            
    """

    def __init__(self, wave='db1'):
        super().__init__()
        wave = pywt.Wavelet(wave)
        g0_col, g1_col = wave.rec_lo, wave.rec_hi
        g0_row, g1_row = g0_col, g1_col
        # Prepare the filters
        self.g0_col = cp.array(g0_col).astype('float32').reshape((1, 1, -1, 1))
        self.g1_col = cp.array(g1_col).astype('float32').reshape((1, 1, -1, 1))
        self.g0_row = cp.array(g0_row).astype('float32').reshape((1, 1, 1, -1))
        self.g1_row = cp.array(g1_row).astype('float32').reshape((1, 1, 1, -1))

    def apply(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass array of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass arrays of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        """
        yl, yh = coeffs
        # Batch the two independent sfb1d(dim=2) calls into one C=2 call,
        # doubling GPU utilisation and halving kernel launches.
        lo_hi = sfb1d(cp.concatenate([yl,        yh[:, :, 1]], axis=1),
                      cp.concatenate([yh[:, :, 0], yh[:, :, 2]], axis=1),
                      self.g0_col, self.g1_col, dim=2)   # [B, 2, H, W']
        yl = sfb1d(lo_hi[:, :1], lo_hi[:, 1:], self.g0_row, self.g1_row, dim=3)
        return yl

# Both caches hand back device arrays, so the current device is part of the
# key -- nothing here switches devices today, but a cached array from another
# one would be a confusing failure if that ever changed.

@lru_cache(maxsize=64)
def _fw_damp_dev(my, sigma, device):
    myr = my // 2 + 1
    y_hat = cp.fft.ifftshift((cp.arange(-my, my, 2) + 1) / 2)[:myr]
    return -cp.expm1(-y_hat**2 / (2 * sigma**2))


def _fw_damp(my, sigma):
    """Münch damping profile for a band of `my` angles, as an rfft mask.

    Cached: it only depends on the band length, so rebuilding it once per
    level per call is pure launch overhead, and at deep levels -- where the
    band is a few tens of samples -- that overhead was most of the FFT stage.
    """
    return _fw_damp_dev(my, sigma, cp.cuda.runtime.getDevice())


@lru_cache(maxsize=16)
def _fw_transforms_dev(wname, device):
    return DWTForward(wave=wname), DWTInverse(wave=wname)


def _fw_transforms(wname):
    """The filter banks for a wavelet; they hold only filter taps, so share them."""
    return _fw_transforms_dev(wname, cp.cuda.runtime.getDevice())


def remove_stripe_fw(data, sigma, wname, level):
    """Remove stripes with wavelet filtering"""

    [nproj, nz, ni] = data.shape

    nproj_pad = nproj + nproj // 8

    # Accepts all wave types available to PyWavelets
    xfm, ifm = _fw_transforms(wname)

    # Wavelet decomposition.
    cc = []
    sli = cp.zeros([nz, 1, nproj_pad, ni], dtype='float32')

    sli[:, 0, (nproj_pad - nproj)//2:(nproj_pad + nproj) //
        2] = data.astype('float32').swapaxes(0, 1)
    for k in range(level):
        sli, c = xfm.apply(sli)
        cc.append(c)
        # FFT – use rfft (real input → ~2× faster, half memory)
        band = cc[k][:, 0, 1]
        _, my, mx = band.shape
        fcV = cp.fft.rfft(band, axis=1)          # [nz, my//2+1, mx]
        fcV *= _fw_damp(my, float(sigma))[:, None]
        cc[k][:, 0, 1] = cp.fft.irfft(fcV, my, axis=1)  # always real

    # Wavelet reconstruction.
    for k in range(level)[::-1]:
        shape0 = cc[k][0, 0, 1].shape
        sli = sli[:, :, :shape0[0], :shape0[1]]
        sli = ifm.apply((sli, cc[k]))

    data = sli[:, 0, (nproj_pad - nproj)//2:(nproj_pad + nproj) //
               2, :ni].astype(data.dtype)  # modified
    data = data.swapaxes(0, 1)

    return data

######## Titarenko ring removal ################
def remove_stripe_ti(data, beta, mask_size):
    """Remove stripes with a new method by V. Titareno """
    gamma = beta*((1-beta)/(1+beta)
                  )**cp.abs(cp.fft.fftfreq(data.shape[-1])*data.shape[-1])
    gamma[0] -= 1
    v = cp.mean(data, axis=0)
    v = v-v[:, 0:1]
    v = cp.fft.irfft(cp.fft.rfft(v)*cp.fft.rfft(gamma))
    mask = cp.zeros(v.shape, dtype=v.dtype)
    mask_size = mask_size*mask.shape[1]
    mask[:, mask.shape[1]//2-mask_size//2:mask.shape[1]//2+mask_size//2] = 1
    data[:] += v*mask
    return data

######## Optimized version for Vo-all ring removal in tomopy#########
def _mpolyfit(x,y):
    n= len(x)
    x_mean = cp.mean(x)
    y_mean = cp.mean(y)
    
    Sxy = cp.sum(x*y) - n*x_mean*y_mean
    Sxx = cp.sum(x*x) - n*x_mean*x_mean
    
    slope = Sxy / Sxx
    intercept = y_mean - slope*x_mean
    return slope,intercept

def _detect_stripe(listdata, snr):
    """
    Algorithm 4 in :cite:`Vo:18`. Used to locate stripes.
    """
    numdata = len(listdata)
    listsorted = cp.sort(listdata)[::-1]
    xlist = cp.arange(0, numdata, 1.0)
    ndrop = cp.int16(0.25 * numdata)
    # (_slope, _intercept) = cp.polyfit(xlist[ndrop:-ndrop - 1],
                                    #   listsorted[ndrop:-ndrop - 1], 1)
    (_slope, _intercept) = _mpolyfit(xlist[ndrop:-ndrop - 1], listsorted[ndrop:-ndrop - 1])

    numt1 = _intercept + _slope * xlist[-1]
    noiselevel = cp.abs(numt1 - _intercept)
    noiselevel = cp.clip(noiselevel, 1e-6, None)
    val1 = cp.abs(listsorted[0] - _intercept) / noiselevel
    val2 = cp.abs(listsorted[-1] - numt1) / noiselevel
    listmask = cp.zeros_like(listdata)
    if (val1 >= snr):
        upper_thresh = _intercept + noiselevel * snr * 0.5
        listmask[listdata > upper_thresh] = 1.0
    if (val2 >= snr):
        lower_thresh = numt1 - noiselevel * snr * 0.5
        listmask[listdata <= lower_thresh] = 1.0
    return listmask

def _detect_stripe_batch(listdata, snr):
    """Batched version of _detect_stripe for 2D input [nz, ni]."""
    nz, numdata = listdata.shape
    listsorted = cp.sort(listdata, axis=1)[:, ::-1]
    xlist = cp.arange(numdata, dtype='float32')
    ndrop = int(0.25 * numdata)
    x = xlist[ndrop:-ndrop - 1]
    y = listsorted[:, ndrop:-ndrop - 1]
    n = x.size
    xm = float(x.mean())
    ym = y.mean(axis=1)                               # [nz]
    Sxy = (y * x).sum(axis=1) - n * ym * xm          # [nz]
    Sxx = float((x**2).sum() - n * xm**2)
    slope = Sxy / Sxx                                  # [nz]
    intercept = ym - slope * xm                        # [nz]
    numt1 = intercept + slope * xlist[-1]              # [nz]
    noiselevel = cp.clip(cp.abs(numt1 - intercept), 1e-6, None)  # [nz]
    val1 = cp.abs(listsorted[:, 0]  - intercept) / noiselevel    # [nz]
    val2 = cp.abs(listsorted[:, -1] - numt1)     / noiselevel    # [nz]
    listmask = cp.zeros((nz, numdata), dtype='float32')
    upper = (intercept + noiselevel * snr * 0.5)[:, None]        # [nz, 1]
    lower = (numt1    - noiselevel * snr * 0.5)[:, None]         # [nz, 1]
    listmask = cp.where((val1 >= snr)[:, None] & (listdata > upper),  1.0, listmask)
    listmask = cp.where((val2 >= snr)[:, None] & (listdata <= lower), 1.0, listmask)
    return listmask


def _inverse_perm3(ids):
    """O(n) inverse permutation along axis=2 via scatter (put_along_axis).

    Replaces argsort(ids, axis=2) which is O(n log n).
    Valid because matindex[i,j]=j so the permutation tracked by matindex
    is identical to ids itself.

    Falls back to O(n log n) argsort on cupy<13 where put_along_axis is
    unavailable (argsort of a permutation IS its inverse permutation).
    """
    if hasattr(cp, 'put_along_axis'):
        ids2 = cp.empty_like(ids)
        src = cp.broadcast_to(cp.arange(ids.shape[2], dtype=ids.dtype)[None, None, :], ids.shape)
        cp.put_along_axis(ids2, ids, src, axis=2)
        return ids2
    return cp.argsort(ids, axis=2)


def _rs_sort3(tomo, size, dim):
    """Batched _rs_sort for 3D input [nproj, nz, ni]."""
    t = cp.transpose(tomo, (2, 1, 0))                              # [ni, nz, nproj]
    ids = cp.argsort(t, axis=2)
    matsort_vals = cp.take_along_axis(t, ids, axis=2)
    del t
    if dim == 1:
        matsort_vals = median_filter(matsort_vals, (size, 1, 1))
    else:
        matsort_vals = median_filter(matsort_vals, (size, 1, size))
    ids2 = _inverse_perm3(ids)
    del ids
    return cp.transpose(cp.take_along_axis(matsort_vals, ids2, axis=2), (2, 1, 0))


def _rs_large3(tomo, snr, size, drop_ratio=0.1, norm=True):
    """Batched _rs_large for 3D input [nproj, nz, ni]."""
    drop_ratio = max(min(drop_ratio, 0.8), 0)
    nproj, nz, ni = tomo.shape
    ndrop = int(0.5 * drop_ratio * nproj)
    # Single argsort replaces cp.sort + later cp.argsort (same logical axis).
    # Normalization divides each nproj-column by a scalar, preserving sort
    # order, so ids computed on the original tomo is reused after normalization.
    t = cp.transpose(tomo, (2, 1, 0))                              # [ni, nz, nproj]
    ids = cp.argsort(t, axis=2)
    sinosort = cp.transpose(cp.take_along_axis(t, ids, axis=2), (2, 1, 0))  # [nproj, nz, ni]
    del t
    sinosmooth = median_filter(sinosort, (1, 1, size))             # [nproj, nz, ni]
    list1 = cp.mean(sinosort[ndrop:nproj - ndrop], axis=0)        # [nz, ni]
    del sinosort
    list2 = cp.mean(sinosmooth[ndrop:nproj - ndrop], axis=0)      # [nz, ni]
    listfact = list1 / list2                                       # [nz, ni]
    listmask = _detect_stripe_batch(listfact, snr)                # [nz, ni]
    listmask = binary_dilation(
        listmask, iterations=1,
        structure=cp.ones((1, 3), dtype=bool)).astype(listmask.dtype)
    if norm:
        tomo = tomo / listfact[None]
    sinosmooth_t = cp.transpose(sinosmooth, (2, 1, 0))            # [ni, nz, nproj]
    del sinosmooth
    ids2 = _inverse_perm3(ids)                                     # O(n) scatter
    del ids
    sino_corrected = cp.transpose(
        cp.take_along_axis(sinosmooth_t, ids2, axis=2), (2, 1, 0))  # [nproj, nz, ni]
    del sinosmooth_t, ids2
    cp.copyto(tomo, sino_corrected, where=(listmask[None] > 0.0))
    return tomo


def _rs_dead3(tomo, snr, size, norm=True):
    """Batched _rs_dead for 3D input [nproj, nz, ni]."""
    tomo = cp.copy(tomo)
    nproj, nz, ni = tomo.shape
    sinosmooth = uniform_filter1d(tomo, 10, axis=0)
    listdiff = cp.sum(cp.abs(tomo - sinosmooth), axis=0)          # [nz, ni]
    listdiffbck = median_filter(listdiff, (1, size))               # [nz, ni]
    listfact = listdiff / listdiffbck                              # [nz, ni]
    listmask = _detect_stripe_batch(listfact, snr)                 # [nz, ni]
    listmask = binary_dilation(
        listmask, iterations=1,
        structure=cp.ones((1, 3), dtype=bool)).astype(listmask.dtype)
    listmask[:, 0:2] = 0.0
    listmask[:, -2:] = 0.0
    for m in range(nz):
        listx = cp.where(listmask[m] < 1.0)[0]
        listxmiss = cp.where(listmask[m] > 0.0)[0]
        if len(listxmiss) > 0:
            matz = tomo[:, m, listx]
            ids = cp.searchsorted(listx, listxmiss)
            tomo[:, m, listxmiss] = (
                matz[:, ids - 1] +
                (listxmiss - listx[ids - 1]) *
                (matz[:, ids] - matz[:, ids - 1]) /
                (listx[ids] - listx[ids - 1]))
    if norm:
        tomo = _rs_large3(tomo, snr, size)
    return tomo


def remove_all_stripe(tomo,
                      snr=3,
                      la_size=61,
                      sm_size=21,
                      dim=1):
    """
    Remove all types of stripe artifacts from sinogram using Nghia Vo's
    approach :cite:`Vo:18` (combination of algorithm 3,4,5, and 6).

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    snr  : float
        Ratio used to locate large stripes.
        Greater is less sensitive.
    la_size : int
        Window size of the median filter to remove large stripes.
    sm_size : int
        Window size of the median filter to remove small-to-medium stripes.
    dim : {1, 2}, optional
        Dimension of the window.

    Returns
    -------
    ndarray
        Corrected 3D tomographic data.
    """
    tomo = _rs_dead3(tomo, snr, la_size)
    tomo = _rs_sort3(tomo, sm_size, dim)
    return tomo
