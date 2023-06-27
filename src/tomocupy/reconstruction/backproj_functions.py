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

from tomocupy.reconstruction import fourierrec,lprec,linerec
from tomocupy.reconstruction import fbp_filter
import cupy as cp

class BackprojFunctions():
    def __init__(self, cl_conf):

        self.args = cl_conf.args
        self.ni = cl_conf.ni
        self.n = cl_conf.n
        self.nz = cl_conf.nz
        self.ncz = cl_conf.ncz
        self.nproj = cl_conf.nproj
        self.ncproj = cl_conf.ncproj
        self.centeri = cl_conf.centeri
        self.center = cl_conf.center
        self.ne = 4*self.n
        if self.args.dtype == 'float16':
            # power of 2 for float16
            self.ne = 2**int(cp.ceil(cp.log2(self.ne)))

        if self.args.lamino_angle == 0:
            if self.args.reconstruction_algorithm == 'fourierrec':
                self.cl_rec = fourierrec.FourierRec(
                    self.n, self.nproj, self.ncz, cp.array(cl_conf.theta), self.args.dtype)
            elif self.args.reconstruction_algorithm == 'lprec':
                self.centeri += 0.5      # consistence with the Fourier based method
                self.center += 0.5
                self.cl_rec = lprec.LpRec(
                    self.n, self.nproj, self.ncz, cp.array(cl_conf.theta), self.args.dtype)
            elif self.args.reconstruction_algorithm == 'linerec':
                self.cl_rec = linerec.LineRec(
                    cp.array(cl_conf.theta), self.nproj, self.nproj, self.ncz, self.ncz, self.n, self.args.dtype)
            self.cl_filter = fbp_filter.FBPFilter(
                self.ne, self.nproj, self.ncz, self.args.dtype)
        else:
            self.cl_rec = linerec.LineRec(
                cp.array(cl_conf.theta), self.nproj, self.ncproj, self.nz, self.ncz, self.n, self.args.dtype)
            self.cl_filter = fbp_filter.FBPFilter(
                self.ne, self.ncproj, self.nz, self.args.dtype)  # note ncproj,nz!
        self.wfilter = self.cl_filter.calc_filter(self.args.fbp_filter)

    def fbp_filter_center(self, data, sht=0):
        """FBP filtering of projections with applying the rotation center shift wrt to the origin"""

        tmp = cp.pad(
            data, ((0, 0), (0, 0), (self.ne//2-self.n//2, self.ne//2-self.n//2)), mode='edge')
        t = cp.fft.rfftfreq(self.ne).astype('float32')
        w = self.wfilter*cp.exp(-2*cp.pi*1j*t*(-self.center +
                                    sht[:, cp.newaxis]+self.n/2))  # center fix

        # tmp = cp.fft.irfft(
            # w*cp.fft.rfft(tmp, axis=2), axis=2).astype(self.args.dtype)  # note: filter works with complex64, however, it doesnt take much time
        self.cl_filter.filter(tmp, w, cp.cuda.get_current_stream())
        data[:] = tmp[:, :, self.ne//2-self.n//2:self.ne//2+self.n//2]

        return data  # reuse input memory
