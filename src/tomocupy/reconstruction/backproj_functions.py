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

from tomocupy.reconstruction import fourierrec, lprec, linerec
from tomocupy.reconstruction import fbp_filter
from tomocupy.global_vars import args, params
import cupy as cp

# Upper bound on the padded rows the FBP filter transforms in one go. The
# filter holds the padded rows, their spectrum and a cuFFT work area of
# about the same size, all proportional to this. Ordinary scans fit in one
# block; very large ones (e.g. double-FOV with tens of thousands of angles)
# would otherwise need tens of GB for the filter alone.
_FILTER_BLOCK_BYTES = 2 * 2**30


class BackprojFunctions():
    def __init__(self):

        params.ne = 4*params.n

        if args.dtype == 'float16':
            # power of 2 for float16
            params.ne = 2**int(cp.ceil(cp.log2(params.ne)))

        theta = cp.array(params.theta)

        if args.lamino_angle != 0:
            # laminography reconstruction with direct discretization of line integrals
            self.cl_rec = linerec.LineRec(
                theta, params.nproj, params.ncproj, params.nz, params.ncz, params.n, args.dtype)
            nz_filter, ntheta_filter = params.nz, params.ncproj  # note ncproj,nz!
        else:
            # tomography
            if args.reconstruction_algorithm == 'fourierrec':
                self.cl_rec = fourierrec.FourierRec(
                    params.n, params.nproj, params.ncz, theta, args.dtype)
            elif args.reconstruction_algorithm == 'lprec':
                params.centeri += 0.5      # consistence with the Fourier based method
                params.center += 0.5
                self.cl_rec = lprec.LpRec(
                    params.n, params.nproj, params.ncz, theta, args.dtype)
            elif args.reconstruction_algorithm == 'linerec':
                self.cl_rec = linerec.LineRec(
                    theta, params.nproj, params.nproj, params.ncz, params.ncz, params.n, args.dtype)
            nz_filter, ntheta_filter = params.ncz, params.nproj

        # Filter block: all of it if it fits, else whole slices at a time,
        # else equal runs of angles within one slice.
        max_rows = max(1, _FILTER_BLOCK_BYTES //
                       (params.ne * cp.dtype(args.dtype).itemsize))
        if nz_filter * ntheta_filter <= max_rows:
            self._fz, self._fp = nz_filter, ntheta_filter
        elif ntheta_filter <= max_rows:
            self._fz, self._fp = max_rows // ntheta_filter, ntheta_filter
        else:
            nblocks = -(-ntheta_filter // max_rows)
            self._fz, self._fp = 1, -(-ntheta_filter // nblocks)
        self.cl_filter = fbp_filter.FBPFilter(
            params.ne, self._fp, self._fz, args.dtype)

        # calculate the FBP filter with quadrature rules
        self.wfilter = self.cl_filter.calc_filter(args.fbp_filter)
        self.pad = params.ne//2 - params.n//2
        self.t = cp.fft.rfftfreq(params.ne).astype('float32')

        # pre-allocated padded block for fbp_filter_center. The cuFFT plan always
        # runs on the whole block, so rows past a short last block just carry
        # leftovers; zeroing once keeps those finite.
        self._tmp = cp.zeros((self._fz, self._fp, params.ne), dtype=args.dtype)

    def fbp_filter_center(self, data, sht=0):
        """FBP filtering of projections with applying the rotation center shift wrt to the origin"""

        nz, ntheta = data.shape[:2]
        if not isinstance(sht, cp.ndarray):
            sht = cp.full(nz, sht, dtype='float32')
        w = self.wfilter*cp.exp(-2*cp.complex64(cp.pi*1j)*(-params.center +
                                               sht[:nz, cp.newaxis]+params.n/2)*self.t)  # center fix
        stream = cp.cuda.get_current_stream()
        fz, fp = self._fz, self._fp
        for z0 in range(0, nz, fz):
            z1 = min(z0 + fz, nz)
            # the kernel reads one filter row per block slice
            wb = w[z0:z1]
            if z1 - z0 < fz:
                wb = cp.concatenate(
                    [wb, cp.zeros((fz - (z1 - z0), w.shape[1]), dtype=w.dtype)])
            for p0 in range(0, ntheta, fp):
                p1 = min(p0 + fp, ntheta)
                d = data[z0:z1, p0:p1]
                tmp = self._tmp[:z1 - z0, :p1 - p0]
                tmp[:, :, self.pad:self.pad+params.n] = d
                tmp[:, :, :self.pad] = d[:, :, :1]
                tmp[:, :, self.pad+params.n:] = d[:, :, -1:]
                self.cl_filter.filter(self._tmp, wb, stream)
                d[:] = tmp[:, :, self.pad:self.pad+params.n]

        return data  # reuse input memory
