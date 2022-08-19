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
                      2:(nproj_pad + nproj)//2, :ni]).astype(data.dtype) #modified
    data = data.swapaxes(0, 1)

    return data
