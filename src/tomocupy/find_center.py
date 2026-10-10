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
from tomocupy.processing import proc_functions
from tomocupy.global_vars import args, params

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

# Number of grey levels the entropy search scores a reconstruction with.
ENTROPY_BINS = 256


class FindCenter():
    '''
    Find rotation axis by comapring 0 and 180 degrees projection with using SIFT
    '''

    def __init__(self, cl_reader):
        # Set ^C interrupt to abort and deallocate memory on GPU
        signal.signal(signal.SIGINT, utils.signal_handler)
        signal.signal(signal.SIGTERM, utils.signal_handler)

        # init tomo functions
        self.cl_proc_func = proc_functions.ProcFunctions()

        # additional refs
        self.cl_reader = cl_reader

    def find_center(self):
        if args.rotation_axis_method == 'sift':
            center = self.find_center_sift()
        elif args.rotation_axis_method == 'vo':
            center = self.find_center_vo()
        elif args.rotation_axis_method == 'entropy':
            center = self.find_center_entropy()
        return (center*2**args.binning).astype('float32')

    def find_center_range_ai(self, args, img_cache, center_of_rotation_cache, out_dir):
        from tomocupy.ai.inference import bin_inference_pipeline
        return bin_inference_pipeline(args, img_cache, center_of_rotation_cache, out_dir)

    def find_center_ai(self, args, img_cache, center_of_rotation_cache, out_dir):
        from tomocupy.ai.inference import inference_pipeline
        return inference_pipeline(args, img_cache, center_of_rotation_cache, out_dir)

    def read_data_try(self, data_queue, id_slice):

        st_z = id_slice
        end_z = id_slice + 2**args.binning

        self.cl_reader.read_data_chunk_to_queue(
            data_queue, params.ids_proj, st_z, end_z,
            params.st_n, params.end_n, 0, params.in_dtype)

    def find_center_sift(self):
        pairs = literal_eval(args.rotation_axis_pairs)

        flat, dark = self.cl_reader.read_flat_dark(
            params.st_n, params.end_n)

        st_row = args.find_center_start_row
        end_row = args.find_center_end_row
        if end_row == -1:
            end_row = args.end_row
        flat = flat[:, st_row:end_row]
        dark = dark[:, st_row:end_row]

        if pairs[0] == pairs[1]:
            pairs[0] = 0
            pairs[1] = params.nproj-1

        data = self.cl_reader.read_pairs(
            pairs, st_row, end_row, params.st_n, params.end_n)

        data = cp.array(data)
        flat = cp.array(flat)
        dark = cp.array(dark)

        data = self.cl_proc_func.darkflat_correction(data, dark, flat)
        data = self.cl_proc_func.minus_log(data)
        data = data.get()
        shifts, nmatches = _register_shift_sift(
            data[::2], data[1::2, :, ::-1], args.rotation_axis_sift_threshold)
        centers = params.n//2-shifts[:, 1]/2+params.st_n
        log.info(f'Number of matched features {nmatches}')
        log.info(
            f'Found centers for projection pairs {centers}, mean: {np.mean(centers)}')
        log.info(
            f'Vertical misalignment {shifts[:, 0]}, mean: {np.mean(shifts[:, 0])}')
        return np.mean(centers)

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

        step = args.center_search_step
        smin = -args.center_search_width
        smax = args.center_search_width

        data_queue = Queue(1)
        self.read_data_try(data_queue, params.id_slices[0])
        item = data_queue.get()
        # copy to gpu
        data = cp.array(item['data'])
        dark = cp.array(item['dark'])
        flat = cp.array(item['flat'])

        data = self.cl_proc_func.darkflat_correction(data, dark, flat)
        data = self.cl_proc_func.minus_log(data)

        _tomo = data.swapaxes(0, 1)[0]
        # Denoising
        # There's a critical reason to use different window sizes
        # between coarse and fine search.
        _tomo_cs = ndimage.gaussian_filter(_tomo, (3, 1), mode='reflect')
        _tomo_fs = ndimage.gaussian_filter(_tomo, (2, 2), mode='reflect')

        # Coarse and fine searches for finding the rotation center.
        init_cen = _search_coarse(_tomo_cs, smin, smax, ratio, drop)
        fine_cen = _search_fine(_tomo_fs, srad, step,
                                init_cen, ratio, drop)
        log.debug('Rotation center search finished: %i', fine_cen)
        return fine_cen

    def find_center_entropy(self):
        """
        Find rotation axis by minimizing the Shannon entropy of the FBP
        reconstruction of one slice (the one selected by --nsino).

        A reconstruction at the right center is sharp, so its grey levels
        concentrate in a few histogram bins and the entropy is low; a wrong
        center smears every feature into a halo and spreads the histogram.
        The slice is reconstructed once per candidate center over the same
        --center-search-width / --center-search-step grid the try mode uses,
        and the lowest-entropy candidate wins.

        All candidates are scored with the SAME ENTROPY_BINS bins over the same
        grey range, otherwise two entropies are not comparable. The range is
        fixed once from the reconstruction at the current center, and values
        outside it are clipped into the end bins rather than dropped -- with
        them dropped the search could lower the entropy simply by pushing mass
        out of the range. Only a centered disc of the field of view, of
        diameter --rotation-axis-entropy-mask times the width, is scored: the
        reconstruction corners are never sampled by all angles, and a bright
        container wall just inside the border would otherwise stretch the grey
        range so far that the sample itself lands in a handful of bins.

        Returns
        -------
        float
            Rotation axis location, in binned pixels.
        """

        # the reconstruction classes are only needed here, and importing them
        # at module level would pull the CUDA kernels into every run
        from tomocupy.reconstruction import backproj_functions

        # candidate centers, same grid and the same convention (center =
        # centeri - shift) as reader.init_sizes_try, so a center reported here
        # means what the same number means in the try reconstruction
        bin = 2**args.binning
        shift_array = np.arange(-args.center_search_width,
                                args.center_search_width +
                                args.center_search_step*bin,
                                args.center_search_step*bin).astype('float32')/bin
        centers = params.centeri - shift_array
        if (args.file_type == 'double_fov') and (params.centeri < params.ni//2):
            # shifts are inverted for the computation, not for the labels
            shift_array = -shift_array

        # one slice, with the preprocessing the reconstruction would do
        data_queue = Queue(1)
        self.read_data_try(data_queue, params.id_slices[0])
        item = data_queue.get()
        data = cp.array(item['data'])
        dark = cp.array(item['dark'])
        flat = cp.array(item['flat'])
        data = self.cl_proc_func.proc_sino(data, dark, flat)
        data = self.cl_proc_func.proc_proj(data)
        data = cp.ascontiguousarray(data.swapaxes(0, 1))  # [1, nproj, n]

        # lprec nudges params.center by 0.5 when its solver is built, and the
        # solver the reconstruction builds later would then do it twice
        center0, centeri0 = params.center, params.centeri
        cl_backproj_func = backproj_functions.BackprojFunctions()

        ncz = params.ncz
        rec = cp.zeros([ncz, params.n, params.n], dtype=params.dtype)
        datat = cp.empty((ncz, data.shape[1], data.shape[2]), dtype=data.dtype)
        sht = cp.zeros(ncz, dtype='float32')
        stream = cp.cuda.get_current_stream()

        def reconstruct(shifts):
            """Reconstruct the slice once per shift; [len(shifts), n, n]."""
            k = len(shifts)
            sht[:k] = cp.asarray(shifts)
            sht[k:] = 0
            datat[:] = data  # the same sinogram for every center
            cl_backproj_func.fbp_filter_center(datat, sht)
            cl_backproj_func.cl_rec.backprojection(rec, datat, stream)
            stream.synchronize()
            return rec[:k]

        try:
            mask = _disc_mask(params.n, args.rotation_axis_entropy_mask)
            # grey range from the reconstruction at the current center
            ref = reconstruct(np.zeros(1, dtype='float32'))[
                0][mask].astype('float32')
            lo, hi = (float(v)
                      for v in cp.percentile(ref, cp.array([0.05, 99.95])))
            del ref
            if not hi > lo:
                log.error('Entropy center search: the reconstruction is flat')
                return np.float32(params.centeri)

            entropies = np.zeros(len(shift_array), dtype='float32')
            for k in range(0, len(shift_array), ncz):
                chunk = shift_array[k:k+ncz]
                utils.printProgressBar(
                    k, len(shift_array), len(shift_array)-k, length=40)
                for i, r in enumerate(reconstruct(chunk)):
                    entropies[k+i] = _entropy(r, mask, lo, hi)
            utils.printProgressBar(
                len(shift_array), len(shift_array), 0, length=40)
        finally:
            params.center, params.centeri = center0, centeri0
            del rec, datat, data, cl_backproj_func
            cp.get_default_memory_pool().free_all_blocks()

        imin = int(np.argmin(entropies))
        center = np.float32(centers[imin])
        if imin == 0 or imin == len(centers)-1:
            log.warning(
                'Entropy minimum is at the edge of the search range, '
                'please extend center-search-width: %g', args.center_search_width)
        log.info(
            f'Entropy {entropies[imin]:.5f} at center {center*bin}, '
            f'{entropies.max():.5f} at worst, {ENTROPY_BINS} bins in [{lo:.4g}, {hi:.4g}]')
        log.debug('Rotation center search finished: %f', center)
        return center


def _disc_mask(n, frac):
    """[n, n] bool: a centered disc of diameter `frac` * n."""

    x = cp.arange(n)-(n-1)/2.0
    return x[:, cp.newaxis]**2+x[cp.newaxis]**2 < (frac*n/2)**2


def _entropy(rec, mask, lo, hi, nbins=ENTROPY_BINS):
    """Shannon entropy, in nats, of the masked grey-level histogram."""

    h = cp.histogram(cp.clip(rec[mask].astype('float32'), lo, hi),
                     bins=nbins, range=(lo, hi))[0].astype('float64')
    p = h/cp.maximum(h.sum(), 1)
    p = p[p > 0]
    return float(-cp.sum(p*cp.log(p)))


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


def _compute_metrics(sino, flip_sino, comp_sino, mask_shifted, list_shift):
    """Compute FFT-based metrics for all shifts using pre-allocated buffers.

    mask_shifted must be pre-ifftshifted so that no fftshift is needed per call.
    mat is pre-allocated: top half holds sino (constant), bottom half is filled
    per iteration without cp.roll or cp.vstack.
    """
    nrow, ncol = sino.shape
    mat = cp.empty((2 * nrow, ncol), dtype=sino.dtype)
    mat[:nrow] = sino
    sino_shift = mat[nrow:]   # writable view — writes go directly into mat
    metrics = []
    for s in list_shift:
        s = float(s)
        if s == int(s):
            si = int(s)
            if si >= 0:
                sino_shift[:, :si] = comp_sino[:, :si]
                sino_shift[:, si:] = flip_sino[:, :ncol - si]
            else:
                sino_shift[:, :ncol + si] = flip_sino[:, -si:]
                sino_shift[:, ncol + si:] = comp_sino[:, ncol + si:]
        else:
            ndimage.shift(flip_sino, (0, s), output=sino_shift, order=3, prefilter=True)
            if s >= 0:
                sino_shift[:, :int(np.ceil(s))] = comp_sino[:, :int(np.ceil(s))]
            else:
                sino_shift[:, int(np.floor(s)):] = comp_sino[:, int(np.floor(s)):]
        metrics.append(cp.mean(cp.abs(cp.fft.fft2(mat)) * mask_shifted))
    return cp.stack(metrics)


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
    mask = cp.fft.ifftshift(_create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop))
    list_shift = 2.0 * (list_cor - cen_fliplr)

    list_metric = _compute_metrics(sino, flip_sino, comp_sino, mask, list_shift)
    minpos = int(cp.argmin(list_metric))
    if minpos == 0:
        log.debug('WARNING!!!Global minimum is out of searching range')
        log.debug('Please extend smin: %i', smin)
    if minpos == len(list_cor) - 1:
        log.debug('WARNING!!!Global minimum is out of searching range')
        log.debug('Please extend smax: %i', smax)
    return list_cor[minpos]


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
    mask = cp.fft.ifftshift(_create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop))
    list_shift = 2.0 * (list_cor - cen_fliplr)
    list_metric = _compute_metrics(sino, flip_sino, comp_sino, mask, list_shift)
    return list_cor[int(cp.argmin(list_metric))]


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
    cen_row = int(np.ceil(nrow / 2.0) - 1)
    cen_col = int(np.ceil(ncol / 2.0) - 1)
    drop = min(drop, int(np.ceil(0.05 * nrow)))
    i = cp.arange(nrow)
    pos = cp.ceil(((i - cen_row) * (dv / radius / du))).astype('int32')
    pos1 = cp.clip(cp.minimum(-pos + cen_col,  pos + cen_col), 0, ncol - 1)
    pos2 = cp.clip(cp.maximum(-pos + cen_col,  pos + cen_col), 0, ncol - 1)
    col = cp.arange(ncol, dtype='int32')
    mask = ((col[None, :] >= pos1[:, None]) & (col[None, :] <= pos2[:, None])).astype('float32')
    mask[cen_row - drop:cen_row + drop + 1, :] = 0.0
    mask[:, cen_col - 1:cen_col + 2] = 0.0
    return mask
