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

from tomocupy import utils
from tomocupy import logging
from tomocupy import conf_sizes
from tomocupy import tomo_functions
from tomocupy import reader
from threading import Thread
import cupy as cp
import numpy as np
import signal
import cv2

__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['FindCenter', ]

log = logging.getLogger(__name__)


class FindCenter():
    '''
    Find rotation axis by comapring 0 and 180 degrees projection with using SIFT
    '''

    def __init__(self, args):
        # Set ^C interrupt to abort and deallocate memory on GPU
        signal.signal(signal.SIGINT, utils.signal_handler)
        signal.signal(signal.SIGTSTP, utils.signal_handler)

        # configure sizes and output files
        cl_reader = reader.Reader(args)
        cl_conf = conf_sizes.ConfSizes(args, cl_reader)

        # init tomo functions
        self.cl_tomo_func = tomo_functions.TomoFunctions(cl_conf, cl_reader)

        # additional refs
        self.args = args
        self.cl_conf = cl_conf
        self.cl_reader = cl_reader

    def find_min_max(self, data):
        """Find min and max values according to histogram"""

        mmin = np.zeros(data.shape[0], dtype='float32')
        mmax = np.zeros(data.shape[0], dtype='float32')

        for k in range(data.shape[0]):
            h, e = np.histogram(data[k][:], 1000)
            stend = np.where(h > np.max(h)*0.005)
            st = stend[0][0]
            end = stend[0][-1]
            mmin[k] = e[st]
            mmax[k] = e[end+1]

        return mmin, mmax

    def register_shift_sift(self, datap1, datap2, th=0.5):
        """Find shifts via SIFT detecting features"""

        mmin, mmax = self.find_min_max(datap1)
        sift = cv2.SIFT_create()
        shifts = np.zeros([datap1.shape[0], 2], dtype='float32')
        for id in range(datap1.shape[0]):
            tmp1 = ((datap2[id]-mmin[id]) /
                    (mmax[id]-mmin[id])*255)
            tmp1[tmp1 > 255] = 255
            tmp1[tmp1 < 0] = 0
            tmp2 = ((datap1[id]-mmin[id]) /
                    (mmax[id]-mmin[id])*255)
            tmp2[tmp2 > 255] = 255
            tmp2[tmp2 < 0] = 0
            # find key points
            tmp1 = tmp1.astype('uint8')
            tmp2 = tmp2.astype('uint8')

            kp1, des1 = sift.detectAndCompute(tmp1, None)
            kp2, des2 = sift.detectAndCompute(tmp2, None)
            # cv2.imwrite('/data/Fister_rec/original_image_right_keypoints.png',cv2.drawKeypoints(tmp1,kp1,None))
            # cv2.imwrite('/data/Fister_rec/original_image_left_keypoints.png',cv2.drawKeypoints(tmp2,kp2,None))
            match = cv2.BFMatcher()
            matches = match.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < th*n.distance:
                    good.append(m)
            if len(good) == 0:
                print('no features found')
                exit()
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=None,
                               flags=2)
            tmp3 = cv2.drawMatches(tmp1, kp1, tmp2, kp2,
                                   good, None, **draw_params)
            #cv2.imwrite("/data/Fister_rec/original_image_drawMatches.jpg", tmp3)
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            shift = (src_pts-dst_pts)[:, 0, :]
            shifts[id] = np.mean(shift, axis=0)[::-1]
        return shifts, len(good)

    def find_center(self):
        from ast import literal_eval
        pairs = literal_eval(self.args.rotation_axis_pairs)

        flat, dark = self.cl_reader.read_flat_dark(
            self.cl_conf.st_n, self.cl_conf.end_n)

        st_row = self.args.find_center_start_row
        end_row = self.args.find_center_end_row
        if end_row == -1:
            end_row = self.args.end_row
        flat = flat[:,st_row:end_row]   
        dark = dark[:,st_row:end_row]  
        
        if pairs[0] == pairs[1]:
            pairs[0] = 0
            pairs[1] = self.cl_conf.nproj-1

        data = self.cl_reader.read_pairs(
            pairs, st_row, end_row, self.cl_conf.st_n, self.cl_conf.end_n)

        data = cp.array(data)
        flat = cp.array(flat)
        dark = cp.array(dark)

        data = self.cl_tomo_func.darkflat_correction(data, dark, flat)
        data = self.cl_tomo_func.minus_log(data)
        data = data.get()
        shifts, nmatches = self.register_shift_sift(
            data[::2], data[1::2, :, ::-1], self.cl_conf.args.rotation_axis_sift_threshold)
        centers = self.cl_conf.n//2-shifts[:, 1]/2+self.cl_conf.st_n
        log.info(f'Number of matched features {nmatches}')
        log.info(
            f'Found centers for projection pairs {centers}, mean: {np.mean(centers)}')
        log.info(
            f'Vertical misalignment {shifts[:, 0]}, mean: {np.mean(shifts[:, 0])}')
        return np.mean(centers)*2**self.args.binning
