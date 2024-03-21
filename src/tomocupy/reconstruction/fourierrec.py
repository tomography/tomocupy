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

from tomocupy import cfunc_fourierrec
from tomocupy import cfunc_fourierrecfp16
import cupy as cp


class FourierRec():
    """Backprojection by the Fourier-based method"""

    def __init__(self, n, nproj, nz, theta, dtype):
        self.theta = theta  # keep theta in memory
        self.nz = nz
        self.n = n
        self.nproj = nproj
        if dtype == 'float16':
            self.fslv = cfunc_fourierrecfp16.cfunc_fourierrec(
                nproj, nz//2, n, self.theta.data.ptr)
        else:
            self.fslv = cfunc_fourierrec.cfunc_fourierrec(
                nproj, nz//2, n, self.theta.data.ptr)

    def backprojection(self, obj, data, stream):
        # reorganize data as a complex array, reuse data
        data = cp.ascontiguousarray(cp.concatenate(
            (data[:self.nz//2, :, :, cp.newaxis], data[self.nz//2:, :, :, cp.newaxis]), axis=3).reshape(data.shape))
        # reuse obj array
        objc = cp.ascontiguousarray(obj.reshape(self.nz//2, self.n, 2*self.n))
        self.fslv.backprojection(obj.data.ptr, data.data.ptr, stream.ptr)
        obj[:] = cp.concatenate((objc[:, :, ::2], objc[:, :, 1::2]))
