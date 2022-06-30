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

from tomocupy_cli import utils
from tomocupy_cli import logging
from tomocupy_cli import conf_io
from tomocupy_cli import tomo_functions
import cupy as cp
import numpy as np
import multiprocessing as mp
import signal


__author__ = "Viktor Nikitin"
__copyright__ = "Copyright (c) 2022, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['GPURec', ]


pinned_memory_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

log = logging.getLogger(__name__)


class GPURec():
    '''
    Class for tomographic reconstruction on GPU with conveyor data processing by chunks.
    Data reading/writing are done in separate processes, CUDA Streams are used to overlap CPU-GPU data transfers with computations.
    The implemented reconstruction method is Fourier-based with exponential functions for interpoaltion in the frequency domain (implemented with CUDA C).
    '''

    def __init__(self, args):
        # Set ^C interrupt to abort and deallocate memory on GPU
        signal.signal(signal.SIGINT, utils.signal_handler)
        signal.signal(signal.SIGTSTP, utils.signal_handler)

        # configure sizes and output files
        cl_conf = conf_io.ConfIO(args)

        # chunks for processing
        self.shape_data_chunk = (cl_conf.nproj, cl_conf.ncz, cl_conf.ni)
        self.shape_recon_chunk = (cl_conf.ncz, cl_conf.n, cl_conf.n)
        self.shape_dark_chunk = (cl_conf.ndark, cl_conf.ncz, cl_conf.ni)
        self.shape_flat_chunk = (cl_conf.nflat, cl_conf.ncz, cl_conf.ni)

        # init tomo functions
        self.cl_tomo_func = tomo_functions.TomoFunctions(cl_conf)

        # streams for overlapping data transfers with computations
        self.stream1 = cp.cuda.Stream(non_blocking=False)
        self.stream2 = cp.cuda.Stream(non_blocking=False)
        self.stream3 = cp.cuda.Stream(non_blocking=False)

        # queue for streaming projections
        self.data_queue = mp.Queue()

        # additional refs
        self.cl_conf = cl_conf

    def recon_all(self):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        # start reading data to a queue
        read_proc = mp.Process(
            target=self.cl_conf.read_data_to_queue, args=(self.data_queue,))
        read_proc.start()

        # refs for faster access
        dtype = self.cl_conf.dtype
        nzchunk = self.cl_conf.nzchunk
        lzchunk = self.cl_conf.lzchunk
        ncz = self.cl_conf.ncz

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, *self.shape_data_chunk], dtype=dtype))
        item_pinned['dark'] = utils.pinned_array(
            np.zeros([2, *self.shape_dark_chunk], dtype=dtype))
        item_pinned['flat'] = utils.pinned_array(
            np.ones([2, *self.shape_flat_chunk], dtype=dtype))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, *self.shape_data_chunk], dtype=dtype)
        item_gpu['dark'] = cp.zeros(
            [2, *self.shape_dark_chunk], dtype=dtype)
        item_gpu['flat'] = cp.ones(
            [2, *self.shape_flat_chunk], dtype=dtype)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_recon_chunk], dtype=dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=dtype)

        # list of procs for parallel writing to hard disk
        write_procs = []

        log.info('Full reconstruction')
        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nzchunk+2):
            utils.printProgressBar(
                k, nzchunk+1, self.data_queue.qsize(), length=40)
            if(k > 0 and k < nzchunk+1):
                with self.stream2:  # reconstruction
                    data = item_gpu['data'][(k-1) % 2]
                    dark = item_gpu['dark'][(k-1) % 2]
                    flat = item_gpu['flat'][(k-1) % 2]
                    rec = rec_gpu[(k-1) % 2]

                    data = self.cl_tomo_func.proc_sino(data, dark, flat)
                    data = self.cl_tomo_func.proc_proj(data)
                    data = cp.ascontiguousarray(data.swapaxes(0, 1))
                    sht = cp.tile(np.float32(0), data.shape[0])
                    data = self.cl_tomo_func.fbp_filter_center(data, sht)
                    self.cl_tomo_func.cl_rec.backprojection(
                        rec, data, self.stream2)

            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    rec_gpu[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < nzchunk):
                # copy to pinned memory
                item = self.data_queue.get()
                item_pinned['data'][k % 2, :, :lzchunk[k]] = item['data']
                item_pinned['dark'][k % 2, :, :lzchunk[k]] = item['dark']
                item_pinned['flat'][k % 2, :, :lzchunk[k]] = item['flat']
                with self.stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])
                    item_gpu['dark'][k % 2].set(item_pinned['dark'][k % 2])
                    item_gpu['flat'][k % 2].set(item_pinned['flat'][k % 2])
            self.stream3.synchronize()
            if(k > 1):
                # add a new thread for writing to hard disk (after gpu->cpu copy is done)
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

    def recon_try(self):
        """GPU reconstruction of 1 slice from an h5file"""

        data, flat, dark = self.cl_conf.read_data_try()
        shift_array = self.cl_conf.shift_array

        data = cp.array(data)
        dark = cp.array(dark)
        flat = cp.array(flat)

        data = self.cl_tomo_func.proc_sino(data, dark, flat)
        data = self.cl_tomo_func.proc_proj(data)
        data = cp.ascontiguousarray(data.swapaxes(0, 1))

        # refs for faster access
        dtype = self.cl_conf.dtype
        nschunk = self.cl_conf.nschunk
        lschunk = self.cl_conf.lschunk
        ncz = self.cl_conf.ncz

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, *self.shape_recon_chunk], dtype=dtype))
        # gpu memory for reconstrution
        rec_gpu = cp.zeros([2, *self.shape_recon_chunk], dtype=dtype)

        # invert shifts for calculations if centeri<ni for double_fov
        write_procs = []
        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nschunk+2):
            utils.printProgressBar(
                k, nschunk+1, self.data_queue.qsize(), length=40)
            if(k > 0 and k < nschunk+1):
                with self.stream2:  # reconstruction
                    sht = cp.pad(cp.array(self.cl_conf.shift_array[(
                        k-1)*ncz:(k-1)*ncz+lschunk[k-1]]), [0, ncz-lschunk[k-1]])
                    datat = cp.tile(data, [ncz, 1, 1])
                    datat = self.cl_tomo_func.fbp_filter_center(datat, sht)
                    self.cl_tomo_func.cl_rec.backprojection(
                        rec_gpu[(k-1) % 2], datat, self.stream2)
            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    rec_gpu[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            self.stream3.synchronize()
            if(k > 1):
                # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                rec_pinned0 = rec_pinned[(k-2) % 2, :lschunk[k-2]].copy()
                for kk in range(lschunk[k-2]):
                    write_proc = mp.Process(
                        target=self.cl_conf.write_data_try, args=(rec_pinned0[kk], self.cl_conf.save_centers[(k-2)*ncz+kk]))
                    write_procs.append(write_proc)
                    write_proc.start()
            self.stream1.synchronize()
            self.stream2.synchronize()

        for proc in write_procs:
            proc.join()
