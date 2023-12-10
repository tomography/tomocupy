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

from tomocupy import cfunc_usfft1d
from tomocupy import cfunc_usfft2d
from tomocupy import cfunc_fft2d

class LamFourierRec():
    """Laminographic backprojection by the Fourier-based method"""

    def __init__(self, n0, n1, n2, ntheta, detw, deth, n1c, nthetac, dethc):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
        self.ntheta = ntheta
        self.detw = detw
        self.deth = deth
        self.n1c = n1c
        self.nthetac = nthetac
        self.dethc = dethc

        self.cl_usfft1d = cfunc_usfft1d.cfunc_usfft1d(n0, n1c, n2, deth)
        self.cl_usfft2d = cfunc_usfft2d.cfunc_usfft2d(
            dethc, n1, n2, ntheta, detw, dethc)
        self.cl_fft2d = cfunc_fft2d.cfunc_fft2d(nthetac, detw, deth)

    def usfft1d_adj(self, out, inp, phi, stream):
        self.cl_usfft1d.adj(out.data.ptr, inp.data.ptr, phi, stream.ptr)

    def usfft2d_adj(self, out, inp, theta, phi, ind, stream):
        self.cl_usfft2d.adj(out.data.ptr, inp.data.ptr,
                            theta.data.ptr, phi, ind, self.deth, stream.ptr)

    def fft2d_fwd(self, out, inp, stream):
        self.cl_fft2d.fwd(out.data.ptr, inp.data.ptr, stream.ptr)
