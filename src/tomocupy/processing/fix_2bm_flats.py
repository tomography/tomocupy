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

"""Correction of moving horizontal flat-field stripes (2-BM mirror deformation).

The mirror stripes in the flat field move/stretch during the scan, so neither the
mean flat nor flat-linear matches the projections. The correction is applied to
projections already divided by the mean flat, T0 = (P - dark) / mean(flat):

1. Dynamic flat field (Van Nieuwenhove et al., Opt. Express 23, 2015): the flat of
   each projection is mean(flat) + sum_k w_k EF_k, with EF_k the leading eigenflats.
   The weights are fitted per projection by minimizing the L1 norm (IRLS) of the
   vertical gradient of the corrected -log image, which is linear in w:
   -log(T0 / (1 + w.E)) ~ -log(T0) + w.E,  E_k = EF_k / mean(flat).
2. Tilted destriping: the residual stripes are estimated in the -log domain as
   median profiles along lines tilted by the stripe angle, in column blocks,
   high-passed in z, interpolated back along the same tilted lines and subtracted.
   Only the 1D profiles are resampled, never the image.

The stripe tilt is estimated from the mean flat by cross-correlating high-passed
row profiles of column blocks.
"""

import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import map_coordinates, gaussian_filter1d

from tomocupy import logging
from tomocupy.global_vars import args

log = logging.getLogger(__name__)

__all__ = ['Fix2bmFlats', 'estimate_tilt']


def _block_profiles(L, nblocks, sigma):
    """High-passed median row profiles of column blocks, (nz, nblocks)."""
    nz, nx = L.shape
    w = nx // nblocks
    prof = cp.median(L[:, :nblocks * w].reshape(nz, nblocks, w), axis=2)
    return prof - gaussian_filter1d(prof, sigma, axis=0, mode='reflect')


def estimate_tilt(flat_mean, nblocks=16, sigma=20, max_angle=3.0):
    """Estimate the stripe tilt angle (degrees) from the mean flat field.

    Returns the angle such that a stripe passing through z0 at the image center
    lies on z = z0 + tan(angle) * (x - xc).
    """
    L = cp.log(cp.maximum(flat_mean, 1))
    nz, nx = L.shape
    w = nx // nblocks
    prof = _block_profiles(L, nblocks, sigma)
    xb = np.arange(nblocks) * w + (w - 1) / 2
    ic = nblocks // 2
    # circular cross-correlation via FFT, taper to reduce edge effects
    taper = cp.hanning(nz)[:, None].astype('float32')
    p = prof * taper
    p = p - p.mean(axis=0)
    p = p / (cp.linalg.norm(p, axis=0) + 1e-12)
    fp = cp.fft.rfft(p, n=2 * nz, axis=0)
    cc = cp.fft.irfft(cp.conj(fp[:, ic:ic + 1]) * fp, n=2 * nz, axis=0).get()  # cc[l, b] = sum p_c(z) p_b(z+l)
    shifts, weights = [], []
    for b in range(nblocks):
        r = int(abs(xb[b] - xb[ic]) * np.tan(np.radians(max_angle))) + 3
        lags = np.arange(-r, r + 1)
        c = cc[lags % (2 * nz), b]
        k = int(np.argmax(c))
        d = 0.0
        if 0 < k < len(c) - 1:
            den = c[k - 1] - 2 * c[k] + c[k + 1]
            if den != 0:
                d = 0.5 * (c[k - 1] - c[k + 1]) / den
        shifts.append(lags[k] + d)
        weights.append(max(c[k], 0) ** 2)
    shifts, weights = np.array(shifts), np.array(weights)
    G = np.stack([xb - xb[ic], np.ones(nblocks)], 1)
    sw = np.sqrt(weights)
    slope = np.linalg.lstsq(G * sw[:, None], shifts * sw, rcond=None)[0][0]
    angle = float(np.degrees(np.arctan(slope)))
    log.info(f'  stripe tilt estimate: {angle:.3f} deg, '
             f'block correlation min/mean {weights.min()**0.5:.2f}/{np.mean(weights**0.5):.2f}')
    if abs(angle) > max_angle or np.mean(weights ** 0.5) < 0.3:
        log.warning(f'  unreliable stripe tilt estimate ({angle:.3f} deg), using 0')
        angle = 0.0
    return angle


class Fix2bmFlats():
    """Dynamic flat field + tilted destriping for projections divided by the mean flat."""

    def __init__(self, flat, dark):
        nblocks = args.fix_2bm_flats_nblocks
        sigma = args.fix_2bm_flats_sigma / 2**args.binning
        dark0 = cp.mean(cp.asarray(dark).astype('float32'), axis=0)
        F = cp.asarray(flat).astype('float32') / args.bright_ratio
        F -= dark0
        nf, nz, nx = F.shape
        M = cp.mean(F, axis=0)
        self.sigma = sigma

        # stripe tilt
        if args.fix_2bm_flats_tilt == 'auto':
            angle = estimate_tilt(M, nblocks, sigma)
        else:
            angle = float(args.fix_2bm_flats_tilt)
            log.info(f'  stripe tilt (given): {angle:.3f} deg')
        a = np.tan(np.radians(angle))

        # eigenflats
        K = min(args.fix_2bm_flats_neigen, nf - 1)
        self.K = max(K, 0)
        if self.K > 0:
            F -= M
            X = F.reshape(nf, -1)
            ev, U = cp.linalg.eigh((X @ X.T).astype('float64'))
            ev, U = ev[::-1], U[:, ::-1]
            rel = (ev / ev[0]).get()
            log.info(f'  eigenflat spectrum (relative): {np.array2string(rel[:min(8, nf)], precision=4)}')
            EF = ((U[:, :self.K].T.astype('float32') @ X) /
                  cp.sqrt(ev[:self.K, None]).astype('float32')).reshape(self.K, nz, nx)
            del X
            valid = M > 0.01 * cp.median(M)
            self.E = cp.where(valid, EF / cp.where(valid, M, 1), 0).astype('float32')
            del EF
            self.dE = cp.diff(self.E, axis=1).reshape(self.K, -1)
        del F

        # tilted sampling coordinates for block profiles and their back-mapping
        w = nx // nblocks
        xc = (nx - 1) / 2
        z = cp.arange(nz, dtype='float32')[:, None]
        x = cp.arange(nblocks * w, dtype='float32')[None, :]
        self.coords = cp.stack([cp.broadcast_to(z + a * (x - xc), (nz, nblocks * w)),
                                cp.broadcast_to(x, (nz, nblocks * w))])
        x = cp.arange(nx, dtype='float32')[None, :]
        bidx = cp.clip((x - (w - 1) / 2) / w, 0, nblocks - 1)
        self.back = cp.stack([cp.broadcast_to(bidx, (nz, nx)),
                              cp.broadcast_to(z - a * (x - xc), (nz, nx))])
        self.nblocks, self.w = nblocks, w

    def _dff_weights(self, L0, niter=4):
        """argmin_w || d/dz (L0 + w.E) ||_1 via iteratively reweighted least squares."""
        g = cp.diff(L0, axis=0).ravel()
        wt = cp.ones_like(g)
        for _ in range(niter):
            Aw = self.dE * wt
            H = Aw @ self.dE.T
            wk = cp.linalg.solve(H + 1e-9 * cp.trace(H) * cp.eye(self.K, dtype=H.dtype), -(Aw @ g))
            wt = 1 / cp.maximum(cp.abs(g + wk @ self.dE), 1e-3)
        return wk

    def _destripe(self, L):
        nz = L.shape[0]
        s = map_coordinates(L, self.coords, order=1, mode='nearest')
        prof = cp.median(s.reshape(nz, self.nblocks, self.w), axis=2).T  # (nblocks, nz)
        prof -= gaussian_filter1d(prof, self.sigma, axis=1, mode='reflect')
        L -= map_coordinates(prof, self.back, order=1, mode='nearest')
        return L

    def correct(self, t0):
        """Correct one projection divided by the mean flat (intensity in, intensity out)."""
        t0 = t0.astype('float32')
        t0 = cp.where(t0 > 0, t0, cp.float32(1))
        L = -cp.log(t0)
        if self.K > 0:
            wk = self._dff_weights(L)
            L += cp.log(cp.maximum(1 + cp.tensordot(wk, self.E, 1), 0.1))
        return cp.exp(-self._destripe(L))

    def correct_chunk(self, data, res):
        """Correct a chunk of projections (nproj, nz, nx)."""
        for k in range(data.shape[0]):
            res[k] = self.correct(data[k])
        return res
