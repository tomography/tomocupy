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
from ast import literal_eval
from queue import Queue
import cupyx.scipy.ndimage as ndimage
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
        self.cl_tomo_func = tomo_functions.TomoFunctions(cl_conf)

        # additional refs
        self.args = args
        self.cl_conf = cl_conf
        self.cl_reader = cl_reader

    def find_center(self):
        if self.args.rotation_axis_method == 'sift':
            center = self.find_center_sift()
        elif self.args.rotation_axis_method == 'vo':
            center = self.find_center_vo()
        return center

    def find_center_sift(self):
        pairs = literal_eval(self.args.rotation_axis_pairs)

        flat, dark = self.cl_reader.read_flat_dark(
            self.cl_conf.st_n, self.cl_conf.end_n)
        if pairs[0] == pairs[1]:
            pairs[0] = 0
            pairs[1] = self.cl_conf.nproj-1

        data = self.cl_reader.read_pairs(
            pairs, self.args.start_row, self.args.end_row, self.cl_conf.st_n, self.cl_conf.end_n)

        data = cp.array(data)
        flat = cp.array(flat)
        dark = cp.array(dark)

        data = self.cl_tomo_func.darkflat_correction(data, dark, flat)
        data = self.cl_tomo_func.minus_log(data)
        data = data.get()
        shifts, nmatches = _register_shift_sift(
            data[::2], data[1::2, :, ::-1], self.cl_conf.args.rotation_axis_sift_threshold)
        centers = self.cl_conf.n//2-shifts[:, 1]/2+self.cl_conf.st_n
        log.info(f'Number of matched features {nmatches}')
        log.info(
            f'Found centers for projection pairs {centers}, mean: {np.mean(centers)}')
        log.info(
            f'Vertical misalignment {shifts[:, 0]}, mean: {np.mean(shifts[:, 0])}')
        return np.mean(centers)

    def read_data_try(self, data_queue, id_slice):
        in_dtype = self.cl_conf.in_dtype
        ids_proj = self.cl_conf.ids_proj
        st_n = self.cl_conf.st_n
        end_n = self.cl_conf.end_n

        st_z = id_slice
        end_z = id_slice + 2**self.args.binning

        self.cl_reader.read_data_chunk_to_queue(
            data_queue, ids_proj, st_z, end_z, st_n, end_n, 0, in_dtype)

    def find_center_sift(self):
        from ast import literal_eval
        pairs = literal_eval(self.args.rotation_axis_pairs)

        flat, dark = self.cl_reader.read_flat_dark(
            self.cl_conf.st_n, self.cl_conf.end_n)

        st_row = self.args.find_center_start_row
        end_row = self.args.find_center_end_row
        if end_row == -1:
            end_row = self.args.end_row
        flat = flat[:, st_row:end_row]
        dark = dark[:, st_row:end_row]

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
        shifts, nmatches = _register_shift_sift(
            data[::2], data[1::2, :, ::-1], self.cl_conf.args.rotation_axis_sift_threshold)
        centers = self.cl_conf.n//2-shifts[:, 1]/2+self.cl_conf.st_n
        log.info(f'Number of matched features {nmatches}')
        log.info(
            f'Found centers for projection pairs {centers}, mean: {np.mean(centers)}')
        log.info(
            f'Vertical misalignment {shifts[:, 0]}, mean: {np.mean(shifts[:, 0])}')
        return np.mean(centers)*2**self.args.binning

    def find_center_vo(self, ind=None, smin=-50, smax=50, srad=6, step=0.25, ratio=0.5, drop=20):
        """
        Find rotation axis location using Nghia Vo's method. :cite:`Vo:14`.
        Parameters
        ----------
        ind : int, optional
            Index of the slice to be used for reconstruction.
        smin, smax : int, optional
            Coarse search radius. Reference to the horizontal center of
            the sinogram.
        srad : float, optional
            Fine search radius.
        step : float, optional
            Step of fine searching.
        ratio : float, optional
            The ratio between the FOV of the camera and the size of object.
            It's used to generate the mask.
        drop : int, optional
            Drop lines around vertical center of the mask.
        Returns
        -------
        float
            Rotation axis location.
        """

        # defaults
        srad = 6  # Fine search radius.
        # The ratio between the FOV of the camera and the size of object. It's used to generate the mask.
        ratio = 0.5
        drop = 20  # Drop lines around vertical center of the mask.

        step = self.args.center_search_step
        smin = -self.args.center_search_width
        smax = self.args.center_search_width

        data_queue = Queue(1)
        self.read_data_try(data_queue, self.cl_conf.id_slices[0])
        item = data_queue.get()
        # copy to gpu
        data = cp.array(item['data'])
        dark = cp.array(item['dark'])
        flat = cp.array(item['flat'])

        data = cp.array(data)
        flat = cp.array(flat)
        dark = cp.array(dark)

        data = self.cl_tomo_func.darkflat_correction(data, dark, flat)
        data = self.cl_tomo_func.minus_log(data)

        _tomo = data.swapaxes(0, 1)[0]
        # Denoising
        # There's a critical reason to use different window sizes
        # between coarse and fine search.
        _tomo_cs = ndimage.gaussian_filter(_tomo, (3, 1), mode='reflect')
        _tomo_fs = ndimage.gaussian_filter(_tomo, (2, 2), mode='reflect')

        # Coarse and fine searches for finding the rotation center.
        # if _tomo.shape[0] * _tomo.shape[1] > 4e6:  # If data is large (>2kx2k)
        #     _tomo_coarse = _downsample(_tomo_cs, level=2)
        #     init_cen = _search_coarse(
        #         _tomo_coarse, smin / 4.0, smax / 4.0, ratio, drop)
        #     fine_cen = _search_fine(_tomo_fs, srad, step,
        #                             init_cen * 4.0, ratio, drop)
        # else:
        init_cen = _search_coarse(_tomo_cs, smin, smax, ratio, drop)
        fine_cen = _search_fine(_tomo_fs, srad, step,
                                    init_cen, ratio, drop)
        log.debug('Rotation center search finished: %i', fine_cen)
        return fine_cen


def _find_min_max(data):
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


def _register_shift_sift(datap1, datap2, th=0.5):
    """Find shifts via SIFT detecting features"""

    mmin, mmax = _find_min_max(datap1)
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
        # cv2.imwrite("/data/Fister_rec/original_image_drawMatches.jpg", tmp3)
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        shift = (src_pts-dst_pts)[:, 0, :]
        shifts[id] = np.mean(shift, axis=0)[::-1]
    return shifts, len(good)


def _downsample(tomo, level):
    for k in range(level):
        tomo = 0.5*(tomo[::2]+tomo[1::2])
        tomo = 0.5*(tomo[:, ::2]+tomo[:, 1::2])
    return tomo


def _calculate_metric(shift_col, sino1, sino2, sino3, mask):
    """
    Metric calculation.
    """
    shift_col = 1.0 * np.squeeze(shift_col)
    if np.abs(shift_col - np.floor(shift_col)) == 0.0:
        shift_col = int(shift_col)
        sino_shift = cp.roll(sino2, shift_col, axis=1)
        if shift_col >= 0:
            sino_shift[:, :shift_col] = sino3[:, :shift_col]
        else:
            sino_shift[:, shift_col:] = sino3[:, shift_col:]
        mat = cp.vstack((sino1, sino_shift))
    else:
        sino_shift = ndimage.shift(
            sino2, (0, shift_col), order=3, prefilter=True)
        if shift_col >= 0:
            shift_int = int(np.ceil(shift_col))
            sino_shift[:, :shift_int] = sino3[:, :shift_int]
        else:
            shift_int = int(np.floor(shift_col))
            sino_shift[:, shift_int:] = sino3[:, shift_int:]
        mat = cp.vstack((sino1, sino_shift))
    metric = cp.mean(
        cp.abs(cp.fft.fftshift(cp.fft.fft2(mat))) * mask)
    return metric


def _search_coarse(sino, smin, smax, ratio, drop):
    """
    Coarse search for finding the rotation center.
    """
    (nrow, ncol) = sino.shape
    cen_fliplr = (ncol - 1.0) / 2.0
    smin = np.int16(np.clip(smin + cen_fliplr, 0, ncol - 1) - cen_fliplr)
    smax = np.int16(np.clip(smax + cen_fliplr, 0, ncol - 1) - cen_fliplr)
    start_cor = ncol // 2 + smin
    stop_cor = ncol // 2 + smax
    flip_sino = cp.fliplr(sino)
    comp_sino = cp.flipud(sino)  # Used to avoid local minima
    list_cor = np.arange(start_cor, stop_cor + 0.5, 0.5)
    list_metric = np.zeros(len(list_cor), dtype=np.float32)
    mask = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)
    list_shift = 2.0 * (list_cor - cen_fliplr)

    for k, s in enumerate(list_shift):
        list_metric[k] = _calculate_metric(s, sino, flip_sino, comp_sino, mask)
    minpos = np.argmin(list_metric)
    if minpos == 0:
        log.debug('WARNING!!!Global minimum is out of searching range')
        log.debug('Please extend smin: %i', smin)
    if minpos == len(list_metric) - 1:
        log.debug('WARNING!!!Global minimum is out of searching range')
        log.debug('Please extend smax: %i', smax)
    cor = list_cor[minpos]
    return cor


def _search_fine(sino, srad, step, init_cen, ratio, drop):
    """
    Fine search for finding the rotation center.
    """
    (nrow, ncol) = sino.shape
    cen_fliplr = (ncol - 1.0) / 2.0
    srad = np.clip(np.abs(srad), 1.0, ncol / 4.0)
    step = np.clip(np.abs(step), 0.1, srad)
    init_cen = np.clip(init_cen, srad, ncol - srad - 1)

    list_cor = init_cen + np.arange(-srad, srad + step, step)

    flip_sino = cp.fliplr(sino)
    comp_sino = cp.flipud(sino)
    mask = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)
    list_shift = 2.0 * (list_cor - cen_fliplr)
    list_metric = np.zeros(len(list_cor), dtype=np.float32)
    for k, s in enumerate(list_shift):
        list_metric[k] = _calculate_metric(s, sino, flip_sino, comp_sino, mask)
    cor = list_cor[np.argmin(list_metric)]
    return cor


def _create_mask(nrow, ncol, radius, drop):
    """
    Make a binary mask to select coefficients outside the double-wedge region.
    Eq.(3) in https://doi.org/10.1364/OE.22.019078
    Parameters
    ----------
    nrow : int
        Image height.
    ncol : int
        Image width.
    radius: int
        Radius of an object, in pixel unit.
    drop : int
        Drop lines around vertical center of the mask.
    Returns
    -------
        2D binary mask.
    """
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * np.pi)
    cen_row = np.int16(np.ceil(nrow / 2.0) - 1)
    cen_col = np.int16(np.ceil(ncol / 2.0) - 1)
    drop = min(drop, np.int16(np.ceil(0.05 * nrow)))
    mask = cp.zeros((nrow, ncol), dtype='float32')
    for i in range(nrow):
        pos = np.int16(np.ceil(((i - cen_row) * dv / radius) / du))
        (pos1, pos2) = np.clip(np.sort(
            (-pos + cen_col, pos + cen_col)), 0, ncol - 1)
        mask[i, pos1:pos2 + 1] = 1.0
    mask[cen_row - drop:cen_row + drop + 1, :] = 0.0
    mask[:, cen_col - 1:cen_col + 2] = 0.0
    return mask
