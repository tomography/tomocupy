# #########################################################################
# Copyright (c) 2022, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2022. UChicago Argonne, LLC. This software was produced       #
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

from tomocupy_cli import fourierrec
from tomocupy_cli import line_summation
from tomocupy_cli import fbp_filter
from tomocupy_cli import retrieve_phase, remove_stripe
from tomocupy_cli import find_rotation
from tomocupy_cli import utils
from tomocupy_cli import logging
from tomocupy_cli import conf_io
from tomocupy_cli import tomo_functions
import multiprocessing as mp
import threading
import cupy as cp
import numpy as np
import numexpr as ne
import signal

__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['GPURecSteps', ]

pinned_memory_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

log = logging.getLogger(__name__)


class GPURecSteps():
    def __init__(self, args):
        # Set ^C interrupt to abort and deallocate memory on GPU
        signal.signal(signal.SIGINT, utils.signal_handler)
        signal.signal(signal.SIGTSTP, utils.signal_handler)

        # configure sizes and output files
        cl_conf = conf_io.ConfIO(args)

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

        # additional refs
        self.cl_conf = cl_conf

    def recon_steps_all(self):
        """GPU reconstruction by loading a full dataset in memory and processing by steps """

        log.info('Step 1. Reading data.')
        data, flat, dark = self.read_data_parallel()

        log.info('Step 2. Processing by chunks in z.')
        data = self.proc_sino_parallel(data, dark, flat)

        log.info('Step 3. Processing by chunks in angles.')
        data = self.proc_proj_parallel(data)

        # Extra block to find centers
        if self.cl_conf.args.reconstruction_algorithm == 'fourierrec':
            log.info('Step 4. Reconstruction by chunks in z.')
            self.recon_sino_parallel(data)
        if self.cl_conf.args.reconstruction_algorithm == 'linesummation':
            log.info('Step 4. Reconstruction by chunks in z and angles.')
            self.recon_sino_proj_parallel(data)

    def recon_steps_try(self):
        """GPU reconstruction of data from an h5file by splitting"""

        log.info('Step 1. Reading data.')
        data, flat, dark = self.read_data_parallel()

        log.info('Step 2. Processing by chunks in z.')
        data = self.proc_sino_parallel(data, dark, flat)

        log.info('Step 3. Processing by chunks in angles.')
        data = self.proc_proj_parallel(data)

        log.info('Step 4. Reconstruction by chunks in center ids and angles.')
        self.recon_try_sino_proj_parallel(data)

    def recon_steps_try_lamino(self):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        log.info('Step 1. Reading data.')
        data, flat, dark = self.read_data_parallel()

        log.info('Step 2. Processing by chunks in z.')
        data = self.proc_sino_parallel(data, dark, flat)

        log.info('Step 3. Processing by chunks in angles.')
        data = self.proc_proj_parallel(data)

        log.info(
            'Step 4. Reconstruction by chunks in lamino angles and projection angles.')
        self.recon_try_lamino_sino_proj_parallel(data)

############################################### Parallel/conveyor execution #############################################

    def read_data_parallel(self, nproc=8):
        """Readin data in parallel (good for ssd disks)"""

        flat, dark = self.cl_conf.read_flat_dark()
        # parallel read of projections
        data = np.zeros([*self.shape_data_full], dtype=self.cl_conf.dtype)
        lchunk = int(np.ceil(data.shape[0]/nproc))
        procs = []
        for k in range(nproc):
            read_proc = threading.Thread(
                target=self.cl_conf.read_data, args=(data, k, lchunk))
            procs.append(read_proc)
            read_proc.start()
        for proc in procs:
            proc.join()

        return data, flat, dark

    def proc_sino_parallel(self, data, dark, flat):

        # refs for faster access
        dtype = self.cl_conf.dtype
        nzchunk = self.cl_conf.nzchunk
        lzchunk = self.cl_conf.lzchunk
        ncz = self.cl_conf.ncz

        res = np.zeros(data.shape, dtype=dtype)

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_z], dtype=dtype))
        item_pinned['dark'] = utils.pinned_array(
            np.zeros([2, *self.shape_dark_chunk_z], dtype=dtype))
        item_pinned['flat'] = utils.pinned_array(
            np.ones([2, *self.shape_flat_chunk_z], dtype=dtype))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros([2, *self.shape_data_chunk_z], dtype=dtype)
        item_gpu['dark'] = cp.zeros([2, *self.shape_dark_chunk_z], dtype=dtype)
        item_gpu['flat'] = cp.ones([2, *self.shape_flat_chunk_z], dtype=dtype)

        # pinned memory for res
        rec_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_z], dtype=dtype))
        # gpu memory for res
        rec_gpu = cp.zeros([2, *self.shape_data_chunk_z], dtype=dtype)

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
                res[:, (k-2)*ncz:(k-2)*ncz+lzchunk[k-2]
                    ] = rec_pinned[(k-2) % 2, :, :lzchunk[k-2]].copy()
            self.stream1.synchronize()
            self.stream2.synchronize()
        return res

    def proc_proj_parallel(self, data):

        # refs for faster access
        dtype = self.cl_conf.dtype
        ntchunk = self.cl_conf.ntchunk
        ltchunk = self.cl_conf.ltchunk
        ncproj = self.cl_conf.ncproj

        res = np.zeros([*self.shape_data_fulln], dtype=dtype)

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_t], dtype=dtype))
        # gpu memory for data item
        data_gpu = cp.zeros([2, *self.shape_data_chunk_t], dtype=dtype)

        # pinned memory for processed data
        rec_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_tn], dtype=dtype))
        # gpu memory for processed data
        rec_gpu = cp.zeros([2, *self.shape_data_chunk_tn], dtype=dtype)

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
        """GPU reconstruction of data from an h5file by splitting into chunks"""
        # refs for faster access
        dtype = self.cl_conf.dtype
        nzchunk = self.cl_conf.nzchunk
        lzchunk = self.cl_conf.lzchunk
        ncz = self.cl_conf.ncz

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_zn], dtype=dtype))

        # gpu memory for data item
        data_gpu = cp.zeros([2, *self.shape_data_chunk_zn], dtype=dtype)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_recon_chunk], dtype=dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=dtype)

        # list of procs for parallel writing to hard disk
        write_procs = []

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
                    rec_gpu[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < nzchunk):
                # copy to pinned memory
                data_pinned[k % 2, :, :lzchunk[k]
                            ] = data[:, k*ncz:k*ncz+lzchunk[k]]
                with self.stream1:  # cpu->gpu copy
                    data_gpu[k % 2].set(data_pinned[k % 2])
            self.stream3.synchronize()
            if(k > 1):
                # add a new proc for writing to hard disk (after gpu->cpu copy is done)
                rec_pinned0 = rec_pinned[(k-2) % 2, :lzchunk[k-2]].copy()
                write_proc = mp.Process(
                    target=self.cl_conf.write_data, args=(rec_pinned0, k-2))
                write_procs.append(write_proc)
                write_proc.start()

            self.stream1.synchronize()
            self.stream2.synchronize()
        # wait until reconstructions are written to hard disk
        for proc in write_procs:
            proc.join()

    def recon_sino_proj_parallel(self, data):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        # refs for faster access
        dtype = self.cl_conf.dtype
        nrchunk = self.cl_conf.nrchunk
        lrchunk = self.cl_conf.lrchunk
        ncz = self.cl_conf.ncz
        ntchunk = self.cl_conf.ntchunk
        ltchunk = self.cl_conf.ltchunk
        ncproj = self.cl_conf.ncproj

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_tn], dtype=dtype))

        # gpu memory for data item
        data_gpu = cp.zeros(
            [2, *self.shape_data_chunk_tn], dtype=dtype)
        theta_gpu = cp.array(self.cl_conf.theta)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_recon_chunk], dtype=dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=dtype)

        # list of procs for parallel writing to hard disk
        write_procs = []

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
                            rec, data0, theta0, self.cl_conf.lamino_angle, (kr-1)*ncz)

                if (kr > 1 and kt == 0):
                    with self.stream3:  # gpu->cpu copy
                        rec_gpu[(kr-2) % 2, :] = rec_gpu[(kr-2) % 2, :, ::-1]
                        rec_gpu[(kr-2) % 2].get(out=rec_pinned[(kr-2) % 2])
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
                    rec_pinned0 = rec_pinned[(kr-2) %
                                             2, :lrchunk[kr-2]].copy()
                    write_proc = mp.Process(
                        target=self.cl_conf.write_data, args=(rec_pinned0, kr-2))
                    write_procs.append(write_proc)
                    write_proc.start()
                self.stream1.synchronize()
                self.stream2.synchronize()
        # wait until reconstructions are written to hard disk
        for proc in write_procs:
            proc.join()

    def recon_try_sino_proj_parallel(self, data):

        # refs for faster access
        dtype = self.cl_conf.dtype
        nschunk = self.cl_conf.nschunk
        lschunk = self.cl_conf.lschunk
        ncz = self.cl_conf.ncz
        ntchunk = self.cl_conf.ntchunk
        ltchunk = self.cl_conf.ltchunk
        ncproj = self.cl_conf.ncproj

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_tn], dtype=dtype))

        # gpu memory for data item
        data_gpu = cp.zeros(
            [2, *self.shape_data_chunk_tn], dtype=dtype)
        theta_gpu = cp.array(self.cl_conf.theta)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_recon_chunk], dtype=dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=dtype)

        write_procs = []
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

                        self.cl_tomo_func.cl_rec.backprojection_try(
                            rec, data0, theta0, self.cl_conf.lamino_angle, self.cl_conf.idslice, sht)

                if (ks > 1 and kt == 0):
                    with self.stream3:  # gpu->cpu copy
                        rec_gpu[(ks-2) % 2] = rec_gpu[(ks-2) % 2, :, ::-1]
                        rec_gpu[(ks-2) % 2].get(out=rec_pinned[(ks-2) % 2])
                if(kt < ntchunk):
                    # copy to pinned memory
                    data_pinned[kt % 2][:ltchunk[kt]
                                        ] = data[kt*ncproj:kt*ncproj+ltchunk[kt]]
                    data_pinned[kt % 2][ltchunk[kt]:] = 0
                    with self.stream1:  # cpu->gpu copy
                        data_gpu[kt % 2].set(data_pinned[kt % 2])
                self.stream3.synchronize()
                if (ks > 1 and kt == 0):
                    rec_pinned0 = rec_pinned[(ks-2) % 2, :lschunk[ks-2]].copy()
                    # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                    for kk in range(lschunk[ks-2]):
                        write_proc = mp.Process(
                            target=self.cl_conf.write_data_try, args=(rec_pinned0[kk], self.cl_conf.save_centers[(ks-2)*ncz+kk]))
                        write_procs.append(write_proc)
                        write_proc.start()
                self.stream1.synchronize()
                self.stream2.synchronize()

    def recon_try_lamino_sino_proj_parallel(self, data):

        # refs for faster access
        dtype = self.cl_conf.dtype
        nschunk = self.cl_conf.nschunk
        lschunk = self.cl_conf.lschunk
        ncz = self.cl_conf.ncz
        ntchunk = self.cl_conf.ntchunk
        ltchunk = self.cl_conf.ltchunk
        ncproj = self.cl_conf.ncproj

        # pinned memory for data item
        data_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk_t], dtype=dtype))

        # gpu memory for data item
        data_gpu = cp.zeros([2, *self.shape_data_chunk_t], dtype=dtype)
        theta_gpu = cp.array(self.cl_conf.theta)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_recon_chunk], dtype=dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=dtype)

        write_procs = []
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
                            rec, data0, theta0, self.cl_conf.lamino_angle, self.cl_conf.idslice, sht)

                if (ks > 1 and kt == 0):
                    with self.stream3:  # gpu->cpu copy
                        rec_gpu[(ks-2) % 2] = rec_gpu[(ks-2) % 2, :, ::-1]
                        rec_gpu[(ks-2) % 2].get(out=rec_pinned[(ks-2) % 2])
                if(kt < ntchunk):
                    # copy to pinned memory
                    data_pinned[kt % 2][:ltchunk[kt]
                                        ] = data[kt*ncproj:kt*ncproj+ltchunk[kt]]
                    data_pinned[kt % 2][ltchunk[kt]:] = 0
                    with self.stream1:  # cpu->gpu copy
                        data_gpu[kt % 2].set(data_pinned[kt % 2])
                self.stream3.synchronize()
                if (ks > 1 and kt == 0):
                    rec_pinned0 = rec_pinned[(ks-2) % 2, :lschunk[ks-2]].copy()
                    # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                    for kk in range(lschunk[ks-2]):
                        write_proc = mp.Process(
                            target=self.cl_conf.write_data_try, args=(rec_pinned0[kk], self.cl_conf.save_centers[(ks-2)*ncz+kk]))
                        write_procs.append(write_proc)
                        write_proc.start()
                self.stream1.synchronize()
                self.stream2.synchronize()
