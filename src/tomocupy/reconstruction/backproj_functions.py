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
            self.cl_filter = fbp_filter.FBPFilter(
                params.ne, params.ncproj, params.nz, args.dtype)  # note ncproj,nz!
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

            self.cl_filter = fbp_filter.FBPFilter(
                params.ne, params.nproj, params.ncz, args.dtype)

        # calculate the FBP filter with quadrature rules
        self.wfilter = self.cl_filter.calc_filter(args.fbp_filter)

    def fbp_filter_center(self, data, sht=0):
        """FBP filtering of projections with applying the rotation center shift wrt to the origin"""

        tmp = cp.pad(
            data, ((0, 0), (0, 0), (params.ne//2-params.n//2, params.ne//2-params.n//2)), mode='edge')
        t = cp.fft.rfftfreq(params.ne).astype('float32')
        w = self.wfilter*cp.exp(-2*cp.pi*1j*t*(-params.center +
                                               sht[:, cp.newaxis]+params.n/2))  # center fix

        self.cl_filter.filter(tmp, w, cp.cuda.get_current_stream())
        data[:] = tmp[:, :, params.ne//2-params.n//2:params.ne//2+params.n//2]

        return data  # reuse input memory
