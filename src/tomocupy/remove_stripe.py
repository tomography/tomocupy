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
import torch
from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
from cupy.fft import (fft, ifft, fft2, ifft2)
from cupyx.scipy.ndimage import median_filter
from cupyx.scipy import signal
from cupyx.scipy.ndimage import binary_dilation
from cupyx.scipy.ndimage import uniform_filter1d
from scipy import interpolate
import numpy as np


def remove_stripe_fw(data, sigma, wname, level):
    """Remove stripes with wavelet filtering"""

    [nproj, nz, ni] = data.shape

    nproj_pad = nproj + nproj // 8
    xshift = int((nproj_pad - nproj) // 2)

    # Accepts all wave types available to PyWavelets
    xfm = DWTForward(J=1, mode='symmetric', wave=wname).cuda()
    ifm = DWTInverse(mode='symmetric', wave=wname).cuda()

    # Wavelet decomposition.
    cc = []
    sli = torch.zeros([nz, 1, nproj_pad, ni], device='cuda')

    sli[:, 0, (nproj_pad - nproj)//2:(nproj_pad + nproj) //
        2] = torch.as_tensor(data.astype('float32').swapaxes(0, 1), device='cuda')
    for k in range(level):
        sli, c = xfm(sli)
        cc.append(c)
        # FFT
        fcV = torch.fft.fft(cc[k][0][:, 0, 1], axis=1)
        _, my, mx = fcV.shape
        # Damping of ring artifact information.
        y_hat = torch.fft.ifftshift((torch.arange(-my, my, 2).cuda() + 1) / 2)
        damp = -torch.expm1(-y_hat**2 / (2 * sigma**2))
        fcV *= torch.transpose(torch.tile(damp, (mx, 1)), 0, 1)
        # Inverse FFT.
        cc[k][0][:, 0, 1] = torch.fft.ifft(fcV, my, axis=1).real

    # Wavelet reconstruction.
    for k in range(level)[::-1]:
        shape0 = cc[k][0][0, 0, 1].shape
        sli = sli[:, :, :shape0[0], :shape0[1]]
        sli = ifm((sli, cc[k]))

    data = cp.asarray(sli[:, 0, (nproj_pad - nproj) //
                      2:(nproj_pad + nproj)//2, :ni]).astype(data.dtype)  # modified
    data = data.swapaxes(0, 1)

    return data


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

# Optimized version for Vo-all ring removal in tomopy
def _rs_sort(sinogram, size, matindex, dim):
    """
    Remove stripes using the sorting technique.
    """
    sinogram = cp.transpose(sinogram)
    matcomb = cp.asarray(cp.dstack((matindex, sinogram)))
    
    # matsort = cp.asarray([row[row[:, 1].argsort()] for row in matcomb])
    ids = cp.argsort(matcomb[:,:,1],axis=1)
    matsort = matcomb.copy()
    matsort[:,:,0] = cp.take_along_axis(matsort[:,:,0],ids,axis=1)
    matsort[:,:,1] = cp.take_along_axis(matsort[:,:,1],ids,axis=1)
    if dim == 1:
        matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, 1))
    else:
        matsort[:, :, 1] = median_filter(matsort[:, :, 1], (size, size))
    
    # matsortback = cp.asarray([row[row[:, 0].argsort()] for row in matsort])
    
    ids = cp.argsort(matsort[:,:,0],axis=1)
    matsortback = matsort.copy()
    matsortback[:,:,0] = cp.take_along_axis(matsortback[:,:,0],ids,axis=1)
    matsortback[:,:,1] = cp.take_along_axis(matsortback[:,:,1],ids,axis=1)
    
    sino_corrected = matsortback[:, :, 1]
    return cp.transpose(sino_corrected)

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

def _rs_large(sinogram, snr, size, matindex, drop_ratio=0.1, norm=True):
    """
    Remove large stripes.
    """
    drop_ratio = max(min(drop_ratio,0.8),0)# = cp.clip(drop_ratio, 0.0, 0.8)
    (nrow, ncol) = sinogram.shape
    ndrop = int(0.5 * drop_ratio * nrow)
    sinosort = cp.sort(sinogram, axis=0)
    sinosmooth = median_filter(sinosort, (1, size))
    list1 = cp.mean(sinosort[ndrop:nrow - ndrop], axis=0)
    list2 = cp.mean(sinosmooth[ndrop:nrow - ndrop], axis=0)
    # listfact = cp.divide(list1,
    #                      list2,
    #                      out=cp.ones_like(list1),
    #                      where=list2 != 0)
    
    listfact = list1/list2
    
    # Locate stripes
    listmask = _detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(listmask.dtype)
    matfact = cp.tile(listfact, (nrow, 1))
    # Normalize
    if norm is True:
        sinogram = sinogram / matfact
    sinogram1 = cp.transpose(sinogram)
    matcombine = cp.asarray(cp.dstack((matindex, sinogram1)))
    
    # matsort = cp.asarray([row[row[:, 1].argsort()] for row in matcombine])
    ids = cp.argsort(matcombine[:,:,1],axis=1)
    matsort = matcombine.copy()
    matsort[:,:,0] = cp.take_along_axis(matsort[:,:,0],ids,axis=1)
    matsort[:,:,1] = cp.take_along_axis(matsort[:,:,1],ids,axis=1)
    
    matsort[:, :, 1] = cp.transpose(sinosmooth)
    # matsortback = cp.asarray([row[row[:, 0].argsort()] for row in matsort])
    ids = cp.argsort(matsort[:,:,0],axis=1)
    matsortback = matsort.copy()
    matsortback[:,:,0] = cp.take_along_axis(matsortback[:,:,0],ids,axis=1)
    matsortback[:,:,1] = cp.take_along_axis(matsortback[:,:,1],ids,axis=1)
    
    sino_corrected = cp.transpose(matsortback[:, :, 1])
    listxmiss = cp.where(listmask > 0.0)[0]
    sinogram[:, listxmiss] = sino_corrected[:, listxmiss]
    return sinogram

def _rs_dead(sinogram, snr, size, matindex, norm=True):
    """
    Remove unresponsive and fluctuating stripes.
    """
    sinogram = cp.copy(sinogram)  # Make it mutable
    (nrow, _) = sinogram.shape
    # sinosmooth = cp.apply_along_axis(uniform_filter1d, 0, sinogram, 10)
    sinosmooth = uniform_filter1d(sinogram, 10, axis=0)
    
    listdiff = cp.sum(cp.abs(sinogram - sinosmooth), axis=0)
    listdiffbck = median_filter(listdiff, size)
    
    
    listfact = listdiff/listdiffbck
    
    listmask = _detect_stripe(listfact, snr)
    listmask = binary_dilation(listmask, iterations=1).astype(listmask.dtype)
    listmask[0:2] = 0.0
    listmask[-2:] = 0.0
    listx = cp.where(listmask < 1.0)[0]
    listy = cp.arange(nrow)
    matz = sinogram[:, listx]
    
    listxmiss = cp.where(listmask > 0.0)[0]    
    
    # finter = interpolate.interp2d(listx.get(), listy.get(), matz.get(), kind='linear')    
    if len(listxmiss) > 0:
        # sinogram_c[:, listxmiss.get()] = finter(listxmiss.get(), listy.get())        
        ids = cp.searchsorted(listx, listxmiss)
        sinogram[:,listxmiss] = matz[:,ids-1]+(listxmiss-listx[ids-1])*(matz[:,ids]-matz[:,ids-1])/(listx[ids]-listx[ids-1])
        
    # Remove residual stripes
    if norm is True:
        sinogram = _rs_large(sinogram, snr, size, matindex)
    return sinogram


def _create_matindex(nrow, ncol):
    """
    Create a 2D array of indexes used for the sorting technique.
    """
    listindex = cp.arange(0.0, ncol, 1.0)
    matindex = cp.tile(listindex, (nrow, 1))
    return matindex

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
    matindex = _create_matindex(tomo.shape[2], tomo.shape[0])
    for m in range(tomo.shape[1]):
        sino = tomo[:, m, :]
        sino = _rs_dead(sino, snr, la_size, matindex)
        sino = _rs_sort(sino, sm_size, matindex, dim)
        tomo[:, m, :] = sino
    return tomo
