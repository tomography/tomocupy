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
from threading import Thread
from tomocupy import reader
from tomocupy import writer
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
    Class for a stepwise tomographic reconstruction on GPU with conveyor data processing by sinogram and projection chunks.
    Steps include 1) pre-processing the whole data volume by splitting into sinograms, 2) pre-processing the whole data volume by splitting into proejections,
    3) reconstructing the whole volume by splitting into sinograms and projections
    The implemented reconstruction methods are 
    1) Fourier-based method with exponential functions for interpoaltion in the frequency domain (implemented with CUDA C),
    2) Direct discretization of the pbackprojection intergral
    """

    def __init__(self, args):
        # Set ^C interrupt to abort and deallocate memory on GPU
        signal.signal(signal.SIGINT, utils.signal_handler)
        signal.signal(signal.SIGTSTP, utils.signal_handler)

        # configure sizes and output files
        cl_reader = reader.Reader(args)
        cl_conf = conf_sizes.ConfSizes(args, cl_reader)
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
        self.cl_tomo_func = tomo_functions.TomoFunctions(cl_conf)

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

    def recon_steps_all(self):
        """GPU reconstruction by loading a full dataset in memory and processing by steps """

        log.info('Step 1. Reading data.')
        data, flat, dark = self.read_data_parallel()        

        if self.args.pre_processing == 'True':
            log.info('Step 2. Processing by chunks in z.')
            data = self.proc_sino_parallel(data, dark, flat)

            log.info('Step 3. Processing by chunks in angles.')
            data = self.proc_proj_parallel(data)
        
        if self.cl_conf.args.reconstruction_type == 'full':
            if self.cl_conf.args.lamino_angle == 0:
                log.info('Step 4. Reconstruction by chunks in z.')
                self.recon_sino_parallel(data)
            else:
                log.info('Step 4. Reconstruction by chunks in z and angles.')
                self.recon_sino_proj_parallel(data)
        elif self.cl_conf.args.reconstruction_type == 'try':
            if self.cl_conf.args.lamino_angle == 0:
                log.info('Step 4. Reconstruction by chunks in center ids.')
                self.recon_try_sino_parallel(data)
            else:
                log.info('Step 4. Reconstruction by chunks in center ids and angles.')
                self.recon_try_sino_proj_parallel(data)
        elif self.cl_conf.args.reconstruction_type == 'try_lamino':
            log.info(
                'Step 4. Reconstruction by chunks in lamino angles and projection angles.')
            self.recon_try_lamino_sino_proj_parallel(data)

############################################### Parallel/conveyor execution #############################################

    def read_data_parallel(self, nthreads=8):
        """Reading data in parallel (good for ssd disks)"""

        st_n = self.cl_conf.st_n
        end_n = self.cl_conf.end_n
        flat, dark = self.cl_reader.read_flat_dark(st_n, end_n)
        # parallel read of projections
        data = np.zeros([*self.shape_data_full], dtype=self.in_dtype)
        lchunk = int(np.ceil(data.shape[0]/nthreads))
        procs = []
        for k in range(nthreads):
            st_proj = k*lchunk
            end_proj = min((k+1)*lchunk,self.args.end_proj-self.args.start_proj)
            if st_proj>=end_proj:
                continue
            read_thread = Thread(
                target=self.cl_reader.read_proj_chunk, args=(data, st_proj, end_proj, self.args.start_row, self.args.end_row, st_n, end_n))
            procs.append(read_thread)
            read_thread.start()
        for proc in procs:
            proc.join()

        return data, flat, dark

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

        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nzchunk+2):
            utils.printProgressBar(
                k, nzchunk+1, nzchunk-k+1, length=40)

            if(k > 0 and k < nzchunk+1):
                with self.stream2:  # reconstruction
                    self.cl_tomo_func.proc_sino(item_gpu['data'][(
                        k-1) % 2], item_gpu['dark'][(k-1) % 2], item_gpu['flat'][(k-1) % 2], rec_gpu[(k-1) % 2])
            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    rec_gpu[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < nzchunk):
                # copy to pinned memory
                item_pinned['data'][k % 2, :, :lzchunk[k]
                                    ] = data[:, k*ncz:k*ncz+lzchunk[k]]
                item_pinned['dark'][k % 2, :, :lzchunk[k]
                                    ] = dark[:, k*ncz:k*ncz+lzchunk[k]]
                item_pinned['flat'][k % 2, :, :lzchunk[k]
                                    ] = flat[:, k*ncz:k*ncz+lzchunk[k]]
                with self.stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])
                    item_gpu['dark'][k % 2].set(item_pinned['dark'][k % 2])
                    item_gpu['flat'][k % 2].set(item_pinned['flat'][k % 2])
            self.stream3.synchronize()
            if(k > 1):
                # copy to result
                res[:, (k-2)*ncz:(k-2)*ncz+lzchunk[k-2]
                    ] = rec_pinned[(k-2) % 2, :, :lzchunk[k-2]].copy()
            self.stream1.synchronize()
            self.stream2.synchronize()
        return res

    def proc_proj_parallel(self, data):
        """Data processing by splitting into projection chunks"""

        # refs for faster access
        ntchunk = self.cl_conf.ntchunk
        ltchunk = self.cl_conf.ltchunk
        ncproj = self.cl_conf.ncproj

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

        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(ntchunk+2):
            utils.printProgressBar(k, ntchunk+1, ntchunk-k+1, length=40)
            if(k > 0 and k < ntchunk+1):
                with self.stream2:  # reconstruction
                    self.cl_tomo_func.proc_proj(
                        data_gpu[(k-1) % 2], rec_gpu[(k-1) % 2])
            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    rec_gpu[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < ntchunk):
                # copy to pinned memory
                data_pinned[k % 2, :ltchunk[k]
                            ] = data[ncproj*k:ncproj*k+ltchunk[k]]
                with self.stream1:  # cpu->gpu copy
                    data_gpu[k % 2].set(data_pinned[k % 2])
            self.stream3.synchronize()
            if(k > 1):
                # add a new proc for writing to hard disk (after gpu->cpu copy is done)
                res[(k-2)*ncproj:(k-2)*ncproj+ltchunk[k-2]
                    ] = rec_pinned[(k-2) % 2, :ltchunk[k-2]].copy()
            self.stream1.synchronize()
            self.stream2.synchronize()
        return res

    def recon_sino_parallel(self, data):
        """Reconstruction by splitting into sinogram chunks"""

        # refs for faster access
        nzchunk = self.cl_conf.nzchunk
        lzchunk = self.cl_conf.lzchunk
        ncz = self.cl_conf.ncz

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_zn], dtype=self.dtype))

        # gpu memory for data item
        data_gpu = cp.zeros([2, *self.shape_data_chunk_zn], dtype=self.dtype)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([self.cl_conf.args.max_write_threads, *self.shape_recon_chunk], dtype=self.dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=self.dtype)

        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nzchunk+2):
            utils.printProgressBar(
                k, nzchunk+1, nzchunk-k+1, length=40)
            if(k > 0 and k < nzchunk+1):
                with self.stream2:  # reconstruction
                    data0 = data_gpu[(k-1) % 2]
                    rec = rec_gpu[(k-1) % 2]
                    data0 = cp.ascontiguousarray(data0.swapaxes(0, 1))
                    data0 = self.cl_tomo_func.fbp_filter_center(
                        data0, cp.tile(np.float32(0), [data0.shape[0], 1]))
                    self.cl_tomo_func.cl_rec.backprojection(
                        rec, data0, self.stream2)
            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    # find free thread
                    ithread = utils.find_free_thread(self.write_threads)
                    rec_gpu[(k-2) % 2].get(out=rec_pinned[ithread])

            if(k < nzchunk):
                # copy to pinned memory
                data_pinned[k % 2, :, :lzchunk[k]
                            ] = data[:, k*ncz:k*ncz+lzchunk[k]]
                with self.stream1:  # cpu->gpu copy
                    data_gpu[k % 2].set(data_pinned[k % 2])
            self.stream3.synchronize()
            if(k > 1):
                # add a new proc for writing to hard disk (after gpu->cpu copy is done)
                st = (k-2)*ncz+self.args.start_row//2**self.args.binning
                end = st+lzchunk[k-2]
                self.write_threads[ithread].run(
                    self.cl_writer.write_data_chunk, (rec_pinned[ithread], st, end, k-2))

            self.stream1.synchronize()
            self.stream2.synchronize()
        # wait until reconstructions are written to hard disk
        for t in self.write_threads:
            t.join()

    def recon_sino_proj_parallel(self, data):
        """Reconstruction by splitting into sinogram and projectionchunks"""

        # refs for faster access
        nrchunk = self.cl_conf.nrchunk
        lrchunk = self.cl_conf.lrchunk
        ncz = self.cl_conf.ncz
        ntchunk = self.cl_conf.ntchunk
        ltchunk = self.cl_conf.ltchunk
        ncproj = self.cl_conf.ncproj

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_tn], dtype=self.dtype))

        # gpu memory for data item
        data_gpu = cp.zeros(
            [2, *self.shape_data_chunk_tn], dtype=self.dtype)
        theta_gpu = cp.array(self.cl_conf.theta)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([self.cl_conf.args.max_write_threads, *self.shape_recon_chunk], dtype=self.dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=self.dtype)

        # Conveyor for data cpu-gpu copy and reconstruction
        for kr in range(nrchunk+2):
            utils.printProgressBar(
                kr, nrchunk+1, nrchunk-kr+1, length=40)
            rec_gpu[(kr-1) % 2][:] = 0
            for kt in range(ntchunk+2):
                if (kr > 0 and kr < nrchunk+1 and kt > 0 and kt < ntchunk+1):
                    with self.stream2:  # reconstruction
                        data0 = data_gpu[(kt-1) % 2]
                        theta0 = theta_gpu[(kt-1)*ncproj:(kt-1)
                                           * ncproj+ltchunk[(kt-1)]]
                        rec = rec_gpu[(kr-1) % 2]

                        data0 = cp.ascontiguousarray(data0.swapaxes(0, 1))
                        data0 = self.cl_tomo_func.fbp_filter_center(
                            data0, cp.tile(np.float32(0), [data0.shape[0], 1]))
                        self.cl_tomo_func.cl_rec.backprojection(
                            rec, data0, self.stream2, theta0, self.cl_conf.lamino_angle, (kr-1)*ncz+self.args.lamino_start_row//2**self.args.binning)

                if (kr > 1 and kt == 0):
                    with self.stream3:  # gpu->cpu copy
                        rec_gpu[(kr-2) % 2, :] = rec_gpu[(kr-2) % 2]
                        ithread = 0
                        while True:
                            if not self.write_threads[ithread].is_alive():
                                break
                            ithread = (
                                ithread+1) % self.cl_conf.args.max_write_threads
                        rec_gpu[(kr-2) % 2].get(out=rec_pinned[ithread])
                if(kt < ntchunk):
                    # copy to pinned memory
                    data_pinned[kt % 2][:ltchunk[kt]
                                        ] = data[kt*ncproj:kt*ncproj+ltchunk[kt]]
                    data_pinned[kt % 2][ltchunk[kt]:] = 0
                    with self.stream1:  # cpu->gpu copy
                        data_gpu[kt % 2].set(data_pinned[kt % 2])
                self.stream3.synchronize()
                if (kr > 1 and kt == 0):
                    # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                    st = (kr-2)*ncz+self.args.lamino_start_row//2**self.args.binning
                    end = st+lrchunk[kr-2]
                    self.write_threads[ithread].run(
                        self.cl_writer.write_data_chunk, (rec_pinned[ithread], st, end, kr-2))

                self.stream1.synchronize()
                self.stream2.synchronize()
        # wait until reconstructions are written to hard disk
        for t in self.write_threads:
            t.join()

    def recon_try_sino_proj_parallel(self, data):
        """Reconstruction of 1 slice with different centers by splitting data into sinogram and projection chunks"""

        # refs for faster access
        nschunk = self.cl_conf.nschunk
        lschunk = self.cl_conf.lschunk
        ncz = self.cl_conf.ncz
        ntchunk = self.cl_conf.ntchunk
        ltchunk = self.cl_conf.ltchunk
        ncproj = self.cl_conf.ncproj

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_tn], dtype=self.dtype))

        # gpu memory for data item
        data_gpu = cp.zeros(
            [2, *self.shape_data_chunk_tn], dtype=self.dtype)
        theta_gpu = cp.array(self.cl_conf.theta)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([self.cl_conf.args.max_write_threads, *self.shape_recon_chunk], dtype=self.dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=self.dtype)


        # Conveyor for data cpu-gpu copy and reconstruction
        for id_slice in self.cl_conf.id_slices:
            log.info(f'Processing slice {id_slice}')
            for ks in range(nschunk+2):
                rec_gpu[(ks-1) % 2][:] = 0

                for kt in range(ntchunk+2):
                    if (ks > 0 and ks < nschunk+1 and kt > 0 and kt < ntchunk+1):
                        with self.stream2:  # reconstruction
                            sht = cp.array(self.cl_conf.shift_array[(
                                ks-1)*ncz:(ks-1)*ncz+lschunk[ks-1]])
                            theta0 = theta_gpu[(kt-1)*ncproj:(kt-1)
                                            * ncproj+ltchunk[(kt-1)]]
                            rec = rec_gpu[(ks-1) % 2]
                            data0 = data_gpu[(kt-1) % 2]
                            data0 = cp.ascontiguousarray(data0.swapaxes(0, 1))
                            data0 = self.cl_tomo_func.fbp_filter_center(
                                data0, cp.tile(np.float32(0), [data0.shape[0], 1]))                            
                            self.cl_tomo_func.cl_rec.backprojection_try(
                                rec, data0, sht, self.stream2, theta0, self.cl_conf.lamino_angle, int(id_slice//2**self.args.binning))

                    if (ks > 1 and kt == 0):
                        with self.stream3:  # gpu->cpu copy
                            rec_gpu[(ks-2) % 2] = rec_gpu[(ks-2) % 2]
                            # find free thread
                            ithread = utils.find_free_thread(self.write_threads)
                            rec_gpu[(ks-2) % 2].get(out=rec_pinned[ithread])
                    if(kt < ntchunk):
                        # copy to pinned memory
                        data_pinned[kt % 2][:ltchunk[kt]
                                            ] = data[kt*ncproj:kt*ncproj+ltchunk[kt]]
                        data_pinned[kt % 2][ltchunk[kt]:] = 0
                        with self.stream1:  # cpu->gpu copy
                            data_gpu[kt % 2].set(data_pinned[kt % 2])
                    self.stream3.synchronize()
                    if (ks > 1 and kt == 0):
                        # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                        for kk in range(lschunk[ks-2]):
                            self.write_threads[ithread].run(self.cl_writer.write_data_try, (
                                rec_pinned[ithread, kk], self.cl_conf.save_centers[(ks-2)*ncz+kk], id_slice))

                    self.stream1.synchronize()
                    self.stream2.synchronize()
            for t in self.write_threads:
                t.join()

    def recon_try_lamino_sino_proj_parallel(self, data):
        """Reconstruction of 1 slice with different lamino angles by splitting data into sinogram and projection chunks"""

        # refs for faster access
        nschunk = self.cl_conf.nschunk
        lschunk = self.cl_conf.lschunk
        ncz = self.cl_conf.ncz
        ntchunk = self.cl_conf.ntchunk
        ltchunk = self.cl_conf.ltchunk
        ncproj = self.cl_conf.ncproj

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_t], dtype=self.dtype))

        # gpu memory for data item
        data_gpu = cp.zeros([2, *self.shape_data_chunk_t], dtype=self.dtype)
        theta_gpu = cp.array(self.cl_conf.theta)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([self.cl_conf.args.max_write_threads, *self.shape_recon_chunk], dtype=self.dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=self.dtype)


        for id_slice in self.cl_conf.id_slices:
            log.info(f'Processing slice {id_slice}')
            # Conveyor for data cpu-gpu copy and reconstruction
            for ks in range(nschunk+2):
                rec_gpu[(ks-1) % 2][:] = 0

                for kt in range(ntchunk+2):
                    if (ks > 0 and ks < nschunk+1 and kt > 0 and kt < ntchunk+1):
                        with self.stream2:  # reconstruction
                            sht = cp.array(self.cl_conf.shift_array[(
                                ks-1)*ncz:(ks-1)*ncz+lschunk[ks-1]])
                            theta0 = theta_gpu[(kt-1)*ncproj:(kt-1)
                                            * ncproj+ltchunk[(kt-1)]]
                            rec = rec_gpu[(ks-1) % 2]
                            data0 = data_gpu[(kt-1) % 2]

                            data0 = cp.ascontiguousarray(data0.swapaxes(0, 1))
                            data0 = self.cl_tomo_func.fbp_filter_center(
                                data0, cp.tile(np.float32(0), [data0.shape[0], 1]))
                            self.cl_tomo_func.cl_rec.backprojection_try_lamino(
                                rec, data0, sht, self.stream2, theta0, self.cl_conf.lamino_angle, int(id_slice//2**self.args.binning))

                    if (ks > 1 and kt == 0):
                        with self.stream3:  # gpu->cpu copy
                            rec_gpu[(ks-2) % 2] = rec_gpu[(ks-2) % 2]
                            # find free thread
                            ithread = utils.find_free_thread(self.write_threads)
                            rec_gpu[(ks-2) % 2].get(out=rec_pinned[ithread])
                    if(kt < ntchunk):
                        # copy to pinned memory
                        data_pinned[kt % 2][:ltchunk[kt]
                                            ] = data[kt*ncproj:kt*ncproj+ltchunk[kt]]
                        data_pinned[kt % 2][ltchunk[kt]:] = 0
                        with self.stream1:  # cpu->gpu copy
                            data_gpu[kt % 2].set(data_pinned[kt % 2])
                    self.stream3.synchronize()
                    if (ks > 1 and kt == 0):
                        # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                        for kk in range(lschunk[ks-2]):
                            self.write_threads[ithread].run(self.cl_writer.write_data_try, (
                                rec_pinned[ithread, kk], self.cl_conf.save_centers[(ks-2)*ncz+kk],id_slice))
                    self.stream1.synchronize()
                    self.stream2.synchronize()
            for t in self.write_threads:
                t.join()

    def recon_try_sino_parallel(self, data):
        """GPU reconstruction of 1 slice for different centers"""

        for id_slice in self.cl_conf.id_slices:
            log.info(f'Processing slice {id_slice}')
            data0 = data[:,id_slice//2**self.args.binning]
            # refs for faster access
            dtype = self.cl_conf.dtype
            nschunk = self.cl_conf.nschunk
            lschunk = self.cl_conf.lschunk
            ncz = self.cl_conf.ncz

            # pinned memory for reconstrution
            rec_pinned = utils.pinned_array(
                np.zeros([self.cl_conf.args.max_write_threads, *self.shape_recon_chunk], dtype=dtype))
            # gpu memory for reconstrution
            rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=dtype)

            # Conveyor for data cpu-gpu copy and reconstruction
            for k in range(nschunk+2):
                utils.printProgressBar(
                    k, nschunk+1, nschunk-k+1, length=40)
                if(k > 0 and k < nschunk+1):
                    with self.stream2:  # reconstruction
                        sht = cp.pad(cp.array(self.cl_conf.shift_array[(
                            k-1)*ncz:(k-1)*ncz+lschunk[k-1]]), [0, ncz-lschunk[k-1]])
                        datat = cp.tile(data0, [ncz, 1, 1])
                        datat = self.cl_tomo_func.fbp_filter_center(datat, sht)
                        self.cl_tomo_func.cl_rec.backprojection(
                            rec_gpu[(k-1) % 2], datat, self.stream2)
                if(k > 1):
                    with self.stream3:  # gpu->cpu copy
                        # find free thread
                        ithread = utils.find_free_thread(self.write_threads)
                        rec_gpu[(k-2) % 2].get(out=rec_pinned[ithread])
                self.stream3.synchronize()
                if(k > 1):
                    # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                    for kk in range(lschunk[k-2]):
                        self.write_threads[ithread].run(self.cl_writer.write_data_try, (
                            rec_pinned[ithread, kk], self.cl_conf.save_centers[(k-2)*ncz+kk],id_slice))

                self.stream1.synchronize()
                self.stream2.synchronize()

            for t in self.write_threads:
                t.join()
                