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
from tomocupy.reconstruction import backproj_parallel
from tomocupy.reconstruction import backproj_lamfourier_parallel
from tomocupy.global_vars import args, params
import signal
import cupy as cp
import numpy as np

__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['GPURecSteps', ]

pinned_memory_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

log = logging.getLogger(__name__)


class GPURecSteps():
    """
    Class for a stepwise tomographic reconstruction on GPU with pipeline data processing by sinogram and projection chunks.
    Steps include 1) pre-processing the whole data volume by splitting into sinograms, 2) pre-processing the whole data volume by splitting into proejections,
    3) reconstructing the whole volume by splitting into sinograms and projections
    The implemented reconstruction methods are 
    1) Fourier-based method with exponential functions for interpoaltion in the frequency domain (implemented with CUDA C),
    2) Direct discretization of the backprojection intergral
    """

    def __init__(self, cl_reader, cl_writer):
        # Set ^C interrupt to abort and deallocate memory on GPU
        signal.signal(signal.SIGINT, utils.signal_handler)
        signal.signal(signal.SIGTERM, utils.signal_handler)

        # use pinned memory
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)

        # chunks for processing
        self.shape_data_chunk_z = (params.nproj, params.ncz, params.ni)
        self.shape_dark_chunk_z = (params.ndark, params.ncz, params.ni)
        self.shape_flat_chunk_z = (params.nflat, params.ncz, params.ni)
        self.shape_data_chunk_zn = (params.nproj, params.ncz, params.n)
        self.shape_data_chunk_t = (params.ncproj, params.nz, params.ni)
        self.shape_data_chunk_tn = (params.ncproj, params.nz, params.n)
        self.shape_recon_chunk = (params.ncz, params.n, params.n)

        # full shapes
        self.shape_data_full = (params.nproj, params.nz, params.ni)
        self.shape_data_fulln = (params.nproj, params.nz, params.n)

        # init tomo functions
        self.cl_proc_func = proc_functions.ProcFunctions()

        # streams for overlapping data transfers with computations
        self.stream1 = cp.cuda.Stream(non_blocking=False)
        self.stream2 = cp.cuda.Stream(non_blocking=False)
        self.stream3 = cp.cuda.Stream(non_blocking=False)

        # threads for data writing to disk
        self.write_threads = []
        for k in range(args.max_write_threads):
            self.write_threads.append(utils.WRThread())

        self.cl_reader = cl_reader
        self.cl_writer = cl_writer

        # define reconstruction method
        if args.lamino_angle != 0 and args.reconstruction_algorithm == 'fourierrec' and args.reconstruction_type == 'full':  # available only for full recon
            self.cl_backproj = backproj_lamfourier_parallel.BackprojLamFourierParallel(
                cl_writer)
        else:
            self.cl_backproj = backproj_parallel.BackprojParallel(cl_writer)

    def recon_steps_all(self):
        """GPU reconstruction by loading a full dataset in memory and processing by steps, with reading the whole data to memory """

        log.info('Reading data.')
        data, flat, dark = self.cl_reader.read_data_parallel()
        if args.pre_processing == 'True':
            log.info('Processing by chunks in z.')
            data = self.proc_sino_parallel(data, dark, flat)
            log.info('Processing by chunks in angles.')
            data = self.proc_proj_parallel(data)
        log.info('Filtered backprojection and writing by chunks.')
        self.cl_backproj.rec_fun(data)

    def proc_sino_parallel(self, data, dark, flat):
        """Data processing by splitting into sinogram chunks"""

        # refs for faster access
        nzchunk = params.nzchunk
        lzchunk = params.lzchunk
        ncz = params.ncz

        # result
        res = np.zeros(data.shape, dtype=params.dtype)

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_z], dtype=params.in_dtype))
        item_pinned['dark'] = utils.pinned_array(
            np.zeros([2, *self.shape_dark_chunk_z], dtype=params.in_dtype))
        item_pinned['flat'] = utils.pinned_array(
            np.ones([2, *self.shape_flat_chunk_z], dtype=params.in_dtype))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, *self.shape_data_chunk_z], dtype=params.in_dtype)
        item_gpu['dark'] = cp.zeros(
            [2, *self.shape_dark_chunk_z], dtype=params.in_dtype)
        item_gpu['flat'] = cp.ones(
            [2, *self.shape_flat_chunk_z], dtype=params.in_dtype)

        # pinned memory for res
        rec_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_z], dtype=params.dtype))
        # gpu memory for res
        rec_gpu = cp.zeros([2, *self.shape_data_chunk_z], dtype=params.dtype)

        # pipeline for data cpu-gpu copy and reconstruction
        for k in range(nzchunk+2):
            utils.printProgressBar(
                k, nzchunk+1, nzchunk-k+1, length=40)

            if (k > 0 and k < nzchunk+1):
                with self.stream2:  # reconstruction
                    self.cl_proc_func.proc_sino(item_gpu['data'][(
                        k-1) % 2], item_gpu['dark'][(k-1) % 2], item_gpu['flat'][(k-1) % 2], rec_gpu[(k-1) % 2])
            if (k > 1):
                with self.stream3:  # gpu->cpu copy
                    rec_gpu[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if (k < nzchunk):
                # copy to pinned memory
                utils.copy(data[:, k*ncz:k*ncz+lzchunk[k]],
                           item_pinned['data'][k % 2, :, :lzchunk[k]])
                utils.copy(dark[:, k*ncz:k*ncz+lzchunk[k]],
                           item_pinned['dark'][k % 2, :, :lzchunk[k]])
                utils.copy(flat[:, k*ncz:k*ncz+lzchunk[k]],
                           item_pinned['flat'][k % 2, :, :lzchunk[k]])

                with self.stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])
                    item_gpu['dark'][k % 2].set(item_pinned['dark'][k % 2])
                    item_gpu['flat'][k % 2].set(item_pinned['flat'][k % 2])
            self.stream3.synchronize()
            if (k > 1):
                # copy to result
                utils.copy(rec_pinned[(k-2) % 2, :, :lzchunk[k-2]],
                           res[:, (k-2)*ncz:(k-2)*ncz+lzchunk[k-2]])
            self.stream1.synchronize()
            self.stream2.synchronize()
        return res

    def proc_proj_parallel(self, data):
        """Data processing by splitting into projection chunks"""

        # refs for faster access
        ntchunk = params.ntchunk
        ltchunk = params.ltchunk
        ncproj = params.ncproj

        if args.file_type != 'double_fov':
            res = data
        else:
            res = np.zeros([*self.shape_data_fulln], dtype=params.dtype)

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_t], dtype=params.dtype))
        # gpu memory for data item
        data_gpu = cp.zeros([2, *self.shape_data_chunk_t], dtype=params.dtype)

        # pinned memory for processed data
        rec_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_tn], dtype=params.dtype))
        # gpu memory for processed data
        rec_gpu = cp.zeros([2, *self.shape_data_chunk_tn], dtype=params.dtype)

        # pipeline for data cpu-gpu copy and reconstruction
        for k in range(ntchunk+2):
            utils.printProgressBar(k, ntchunk+1, ntchunk-k+1, length=40)
            if (k > 0 and k < ntchunk+1):
                with self.stream2:  # reconstruction
                    self.cl_proc_func.proc_proj(
                        data_gpu[(k-1) % 2], 0, self.shape_data_chunk_t[1], res=rec_gpu[(k-1) % 2])
            if (k > 1):
                with self.stream3:  # gpu->cpu copy
                    rec_gpu[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if (k < ntchunk):
                # copy to pinned memory
                utils.copy(data[ncproj*k:ncproj*k+ltchunk[k]],
                           data_pinned[k % 2, :ltchunk[k]])
                with self.stream1:  # cpu->gpu copy
                    data_gpu[k % 2].set(data_pinned[k % 2])
            self.stream3.synchronize()
            if (k > 1):
                utils.copy(rec_pinned[(k-2) % 2, :ltchunk[k-2]],
                           res[(k-2)*ncproj:(k-2)*ncproj+ltchunk[k-2]])
            self.stream1.synchronize()
            self.stream2.synchronize()
        return res
