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

from tomocupy import fourierrec
from tomocupy import lprec
from tomocupy import fbp_filter
from tomocupy import linerec
from tomocupy import utils
from tomocupy import retrieve_phase, remove_stripe, adjust_projections
import cupyx.scipy.ndimage as ndimage


import cupy as cp
import numpy as np


class TomoFunctions():
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
            self.ne = 2**int(np.ceil(np.log2(self.ne)))

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


    def darkflat_correction(self, data, dark, flat):
        """Dark-flat field correction"""

        dark0 = dark.astype(self.args.dtype, copy=False)
        flat0 = flat.astype(self.args.dtype, copy=False)
        # works only for processing all angles
        if self.args.flat_linear == 'True' and data.shape[0] == self.nproj:
            flat0_p0 = cp.mean(flat0[:flat0.shape[0]//2], axis=0)
            flat0_p1 = cp.mean(flat0[flat0.shape[0]//2+1:], axis=0)
            v = cp.linspace(0, 1, self.nproj)[..., cp.newaxis, cp.newaxis]
            flat0 = (1-v)*flat0_p0+v*flat0_p1
        else:
            flat0 = cp.mean(flat0, axis=0)
        dark0 = cp.mean(dark0, axis=0)
        res = (data.astype(self.args.dtype, copy=False)-dark0) / (flat0-dark0+1e-3)
        res[res <= 0] = 1
        return res

    def minus_log(self, data):
        """Taking negative logarithm"""

        data[:] = -cp.log(data)
        data[cp.isnan(data)] = 6.0
        data[cp.isinf(data)] = 0
        return data  # reuse input memory

    def remove_outliers(self, data):
        """Remove outliers"""

        if (int(self.args.dezinger) > 0):
            w = int(self.args.dezinger)
            if len(data.shape) == 3:
                fdata = ndimage.median_filter(data, [w, 1, w])
            else:
                fdata = ndimage.median_filter(data, [w, w])
            data[:] = cp.where(cp.logical_and(
                data > fdata, (data - fdata) > self.args.dezinger_threshold), fdata, data)
        return data
        # if(int(self.args.dezinger) > 0):
        #     r = int(self.args.dezinger)
        #     fdata = ndimage.median_filter(data, [1, r, r])
        #     ids = cp.where(cp.abs(fdata-data) > 0.5*cp.abs(fdata))
        #     data[ids] = fdata[ids]
        # return data


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

    def pad360(self, data):
        """Pad data with 0 to handle 360 degrees scan"""

        if (self.centeri < self.ni//2):
            # if rotation center is on the left side of the ROI
            data[:] = data[:, :, ::-1]
        w = max(1, int(2*(self.ni-self.center)))

        if self.args.pad_endpoint == 'True':
            v = cp.linspace(1, 0, w, endpoint=True)
        else:
            v = cp.linspace(1, 0, w, endpoint=False)
        v = v**5*(126-420*v+540*v**2-315*v**3+70*v**4)
        data[:, :, -w:] *= v

        # double sinogram size with adding 0
        data = cp.pad(data, ((0, 0), (0, 0), (0, data.shape[-1])), 'constant')
        return data

    def proc_sino(self, data, dark, flat, res=None):
        """Processing a sinogram data chunk"""

        if not isinstance(res, cp.ndarray):
            res = cp.zeros(data.shape, self.args.dtype)
        # dark flat field correrction
        data[:] = self.remove_outliers(data)
        dark[:] = self.remove_outliers(dark)
        flat[:] = self.remove_outliers(flat)
        res[:] = self.darkflat_correction(data, dark, flat)
        # remove stripes
        if self.args.remove_stripe_method == 'fw':
            res[:] = remove_stripe.remove_stripe_fw(
                res, self.args.fw_sigma, self.args.fw_filter, self.args.fw_level)
        elif self.args.remove_stripe_method == 'ti':
            res[:] = remove_stripe.remove_stripe_ti(
                res, self.args.ti_beta, self.args.ti_mask)
        elif self.args.remove_stripe_method == 'vo-all':
            res[:] = remove_stripe.remove_all_stripe(
                res, self.args.vo_all_snr, self.args.vo_all_la_size, self.args.vo_all_sm_size, self.args.vo_all_dim)

        return res

    def proc_proj(self, data, res=None):
        """Processing a projection data chunk"""

        if not isinstance(res, cp.ndarray):
            res = cp.zeros(
                [data.shape[0], data.shape[1], self.n], self.args.dtype)
        # retrieve phase
        if self.args.retrieve_phase_method == 'paganin':
            data[:] = retrieve_phase.paganin_filter(
                data,  self.args.pixel_size*1e-4, self.args.propagation_distance/10, self.args.energy, self.args.retrieve_phase_alpha)
        if self.args.rotate_proj_angle != 0:
            data[:] = adjust_projections.rotate(
                data, self.args.rotate_proj_angle, self.args.rotate_proj_order)
        # minus log
        if self.args.minus_log == 'True':
            data[:] = self.minus_log(data)
        # padding for 360 deg recon
        if (self.args.file_type == 'double_fov'):
            res[:] = self.pad360(data)
        else:
            res[:] = data[:]
        return res
