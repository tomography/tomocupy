#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2022, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################


from tomocupy import cfunc_linerec
from tomocupy import cfunc_linerecfp16
import cupy as cp


class LineRec():
    """Backprojection by summation over lines"""

    def __init__(self, theta, nproj, ncproj, nz, ncz, n, dtype):
        self.nproj = nproj
        self.ncproj = ncproj
        self.nz = nz
        self.ncz = ncz
        self.n = n        
        self.dtype = dtype
        self.theta = cp.array(theta)
        
        if dtype == 'float16':
            self.fslv = cfunc_linerecfp16.cfunc_linerec(nproj, nz, n, ncproj, ncz)
        else:
            self.fslv = cfunc_linerec.cfunc_linerec(nproj, nz, n, ncproj, ncz)
        

    def backprojection(self, f, data, stream=0, theta=[], lamino_angle=0, sz=0):
        if len(theta)==0:
            theta = self.theta
            f[:]=0
        phi = cp.float(cp.pi/2+(lamino_angle)/180*cp.pi)
        self.fslv.backprojection(f.data.ptr, data.data.ptr, theta.data.ptr, phi, sz, stream.ptr)
        
    def backprojection_try(self, f, data, sh, stream=0, theta=[], lamino_angle=0, sz=0):
        if len(theta)==0:
            theta = self.theta
        phi = cp.float(cp.pi/2+(lamino_angle)/180*cp.pi)
        self.fslv.backprojection_try(f.data.ptr, data.data.ptr, theta.data.ptr, sh.data.ptr, phi, sz, stream.ptr)

    def backprojection_try_lamino(self, f, data, sh, stream=0, theta=[], lamino_angle=0, sz=0):
        if len(theta)==0:
            theta = self.theta
        phi = (cp.pi/2+(lamino_angle+sh)/180*cp.pi).astype('float32')
        self.fslv.backprojection_try_lamino(f.data.ptr, data.data.ptr, theta.data.ptr, phi.data.ptr, sz, stream.ptr)
