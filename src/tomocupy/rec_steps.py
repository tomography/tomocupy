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
from tomocupy import config_sizes
from tomocupy import reader
from tomocupy import writer
from tomocupy.processing import proc_functions
from tomocupy.reconstruction import backproj_parallel
from tomocupy.reconstruction import backproj_lamfourier_parallel

from threading import Thread
import signal
import cupy as cp
import numpy as np

__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['GPURecSteps', ]

log = logging.getLogger(__name__)


class GPURecSteps():
    """
    Class for a stepwise tomographic reconstruction on GPU with pipeline data processing by sinogram and projection chunks.
    Steps include 1) pre-processing the whole data volume by splitting into sinograms, 2) pre-processing the whole data volume by splitting into proejections,
    3) reconstructing the whole volume by splitting into sinograms and projections
    """

    def __init__(self, args):
        # Set ^C interrupt to abort and deallocate memory on GPU
        signal.signal(signal.SIGINT, utils.signal_handler)
        signal.signal(signal.SIGTERM, utils.signal_handler)

        # use pinned memory
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
        # configure sizes and output files
        cl_reader = reader.Reader(args)
        cl_conf = config_sizes.ConfigSizes(args, cl_reader)
        cl_writer = writer.Writer(args, cl_conf)

        # chunks for processing
        self.shape_data_chunk_z = (cl_conf.nproj, cl_conf.ncz, cl_conf.ni)
        self.shape_dark_chunk_z = (cl_conf.ndark, cl_conf.ncz, cl_conf.ni)
        self.shape_flat_chunk_z = (cl_conf.nflat, cl_conf.ncz, cl_conf.ni)
        self.shape_data_chunk_zn = (cl_conf.nproj, cl_conf.ncz, cl_conf.n)
        self.shape_data_chunk_t = (cl_conf.ncproj, cl_conf.nz, cl_conf.ni)
        self.shape_data_chunk_tn = (cl_conf.ncproj, cl_conf.nz, cl_conf.n)
        self.shape_recon_chunk = (cl_conf.ncz, cl_conf.n, cl_conf.n)

        # full shapes
        self.shape_data_full = (cl_conf.nproj, cl_conf.nz, cl_conf.ni)
        self.shape_data_fulln = (cl_conf.nproj, cl_conf.nz, cl_conf.n)

        # init tomo functions
        self.cl_proc_func = proc_functions.ProcFunctions(cl_conf)

        # streams for overlapping data transfers with computations
        self.stream1 = cp.cuda.Stream(non_blocking=False)
        self.stream2 = cp.cuda.Stream(non_blocking=False)
        self.stream3 = cp.cuda.Stream(non_blocking=False)

        # threads for data writing to disk
        self.write_threads = []
        for k in range(cl_conf.args.max_write_threads):
            self.write_threads.append(utils.WRThread())

        # additional refs
        self.dtype = cl_conf.dtype
        self.in_dtype = cl_conf.in_dtype
        self.args = args
        self.cl_conf = cl_conf
        self.cl_reader = cl_reader
        self.cl_writer = cl_writer

        # define reconstruction method
        if self.cl_conf.args.lamino_angle != 0 and self.args.reconstruction_algorithm == 'fourierrec' and self.args.reconstruction_type == 'full':  # available only for full recon
            self.cl_backproj = backproj_lamfourier_parallel.BackprojLamFourierParallel(
                cl_conf, cl_writer)
        else:
            self.cl_backproj = backproj_parallel.BackprojParallel(
                cl_conf, cl_writer)

    def read_data_parallel(self, nthreads=16):
        """Reading data in parallel (good for ssd disks)"""

        st_n = self.cl_conf.st_n
        end_n = self.cl_conf.end_n
        flat, dark = self.cl_reader.read_flat_dark(st_n, end_n)
        # parallel read of projections
        data = np.zeros([*self.shape_data_full], dtype=self.in_dtype)
        lchunk = int(np.ceil(data.shape[0]/nthreads))
        read_threads = []
        for k in range(nthreads):
            st_proj = k*lchunk
            end_proj = min((k+1)*lchunk, self.args.end_proj -
                           self.args.start_proj)
            if st_proj >= end_proj:
                continue
            read_thread = Thread(
                target=self.cl_reader.read_proj_chunk, args=(data, st_proj, end_proj, self.args.start_row, self.args.end_row, st_n, end_n))
            read_threads.append(read_thread)
            read_thread.start()
        # wait for threads
        for read_thread in read_threads:
            read_thread.join()

        return data, flat, dark

    def recon_steps_all(self):
        """GPU reconstruction by loading a full dataset in memory and processing by steps, with reading the whole data to memory """

        log.info('Reading data.')
        data, flat, dark = self.read_data_parallel()
        if self.args.pre_processing == 'True':
            log.info('Processing by chunks in z.')
            data = self.proc_sino_parallel(data, dark, flat)
            log.info('Processing by chunks in angles.')
            data = self.proc_proj_parallel(data)

        log.info('Filtered backprojection and writing by chunks.')
        self.cl_backproj.rec_fun(data)

    def proc_sino_parallel(self, data, dark, flat):
        """Data processing by splitting into sinogram chunks"""

        # refs for faster access
        nzchunk = self.cl_conf.nzchunk
        lzchunk = self.cl_conf.lzchunk
        ncz = self.cl_conf.ncz

        # result
        res = np.zeros(data.shape, dtype=self.dtype)

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_z], dtype=self.in_dtype))
        item_pinned['dark'] = utils.pinned_array(
            np.zeros([2, *self.shape_dark_chunk_z], dtype=self.in_dtype))
        item_pinned['flat'] = utils.pinned_array(
            np.ones([2, *self.shape_flat_chunk_z], dtype=self.in_dtype))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, *self.shape_data_chunk_z], dtype=self.in_dtype)
        item_gpu['dark'] = cp.zeros(
            [2, *self.shape_dark_chunk_z], dtype=self.in_dtype)
        item_gpu['flat'] = cp.ones(
            [2, *self.shape_flat_chunk_z], dtype=self.in_dtype)

        # pinned memory for res
        rec_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_z], dtype=self.dtype))
        # gpu memory for res
        rec_gpu = cp.zeros([2, *self.shape_data_chunk_z], dtype=self.dtype)

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
        ntchunk = self.cl_conf.ntchunk
        ltchunk = self.cl_conf.ltchunk
        ncproj = self.cl_conf.ncproj

        if self.args.file_type != 'double_fov':
            res = data
        else:
            res = np.zeros([*self.shape_data_fulln], dtype=self.dtype)

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_t], dtype=self.dtype))
        # gpu memory for data item
        data_gpu = cp.zeros([2, *self.shape_data_chunk_t], dtype=self.dtype)

        # pinned memory for processed data
        rec_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_tn], dtype=self.dtype))
        # gpu memory for processed data
        rec_gpu = cp.zeros([2, *self.shape_data_chunk_tn], dtype=self.dtype)

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
