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
from tomocupy_cli import fbp_filter
from tomocupy_cli import remove_stripe
from tomocupy_cli import utils
from tomocupy_cli import logging
from tomocupy_cli import confio
import cupy as cp
import numpy as np
import multiprocessing as mp
import signal
from tomocupy_cli import find_rotation


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

        cl_conf = confio.ConfIO(args)

        self.n = cl_conf.n
        self.nz = cl_conf.nz
        self.ncz = cl_conf.ncz
        self.nproj = cl_conf.nproj
        self.center = cl_conf.center
        self.ni = cl_conf.ni
        self.centeri = cl_conf.centeri
        self.ndark = cl_conf.ndark
        self.nflat = cl_conf.nflat
        self.ids_proj = cl_conf.ids_proj
        self.nzchunk = cl_conf.nzchunk
        self.lzchunk = cl_conf.lzchunk
        self.nschunk = cl_conf.nschunk
        self.lschunk = cl_conf.lschunk

        self.args = cl_conf.args
        self.cl_conf = cl_conf

        theta = cp.array(cl_conf.theta)
        self.cl_filter = fbp_filter.FBPFilter(
            self.n, self.nproj, self.ncz, args.dtype)
        self.cl_rec = fourierrec.FourierRec(
            self.n, self.nproj, self.ncz, theta, args.dtype)

        # queue for streaming projections
        self.data_queue = mp.Queue()

    def darkflat_correction(self, data, dark, flat):
        """Dark-flat field correction"""

        dark0 = cp.mean(dark.astype(self.args.dtype), axis=0)
        flat0 = cp.mean(flat.astype(self.args.dtype), axis=0)
        data = (data.astype(self.args.dtype)-dark0)/(flat0-dark0)
        return data

    def minus_log(self, data):
        """Taking negative logarithm"""

        data = -cp.log(data)
        data[cp.isnan(data)] = 6.0
        data[cp.isinf(data)] = 0
        return data

    def fbp_filter_center(self, data, sh=0):
        """FBP filtering of projections"""

        ne = 3*self.n//2
        if self.args.dtype == 'float16':
            # power of 2 for float16
            ne = 2**int(np.ceil(np.log2(3*self.n//2)))
        t = cp.fft.rfftfreq(ne).astype('float32')
        if self.args.gridrec_filter == 'parzen':
            w = t * (1 - t * 2)**3
        elif self.args.gridrec_filter == 'shepp':
            w = t * cp.sinc(t)
        sht = cp.zeros([self.ncz, 1], dtype='float32')

        if isinstance(sh, cp.ndarray):
            # try
            data = cp.tile(data, (self.ncz//2, 1, 1))
            sht[:len(sh)] = sh.reshape(len(sh), 1)
        else:
            # full
            sht[:] = sh
        w = w*cp.exp(-2*cp.pi*1j*t*(-self.center+sht+self.n/2))  # center fix
        data = cp.pad(
            data, ((0, 0), (0, 0), (ne//2-self.n//2, ne//2-self.n//2)), mode='edge')
        self.cl_filter.filter(data, w, cp.cuda.get_current_stream())
        # data = cp.fft.irfft(
        # w[:,cp.newaxis]*cp.fft.rfft(data, axis=2), axis=2).astype(self.args.dtype)  # note: filter works with complex64, however, it doesnt take much time
        data = data[:, :, ne//2-self.n//2:ne//2+self.n//2]

        return data

    def pad360(self, data):
        """Pad data with 0 to handle 360 degrees scan"""

        if(self.centeri < self.ni//2):
            # if rotation center is on the left side of the ROI
            data[:] = data[:, :, ::-1]
        w = max(1, int(2*(self.ni-self.center)))
        v = cp.linspace(1, 0, w, endpoint=False)
        v = v**5*(126-420*v+540*v**2-315*v**3+70*v**4)
        data[:, :, -w:] *= v

        # double sinogram size with adding 0
        data = cp.pad(data, ((0, 0), (0, 0), (0, data.shape[-1])), 'constant')
        return data

    def recon(self, obj, data, dark, flat):
        """Full reconstruction pipeline for a data chunk"""

        # dark-flat field correction
        data = self.darkflat_correction(data, dark, flat)
        # minus log
        data = self.minus_log(data)
        # remove stripes
        if(self.args.remove_stripe_method == 'fw'):
            data = remove_stripe.remove_stripe_fw(
                data, self.args.fw_sigma, self.args.fw_filter, self.args.fw_level)
        # padding for 360 deg recon
        if(self.args.file_type == 'double_fov'):
            data = self.pad360(data)
        # fbp filter and compensatio for the center
        data = self.fbp_filter_center(data)
        # reshape to sinograms
        data = cp.ascontiguousarray(data.swapaxes(0, 1))
        # backprojection
        self.cl_rec.backprojection(obj, data, cp.cuda.get_current_stream())

    def recon_all(self):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        # start reading data to a queue
        read_proc = mp.Process(
            target=self.cl_conf.read_data_to_queue, args=(self.data_queue,))
        read_proc.start()

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.nproj, self.ncz, self.ni], dtype=self.args.dtype))
        item_pinned['dark'] = utils.pinned_array(
            np.zeros([2, self.ndark, self.ncz, self.ni], dtype=self.args.dtype))
        item_pinned['flat'] = utils.pinned_array(
            np.ones([2, self.nflat, self.ncz, self.ni], dtype=self.args.dtype))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.nproj, self.ncz, self.ni], dtype=self.args.dtype)
        item_gpu['dark'] = cp.zeros(
            [2, self.ndark, self.ncz, self.ni], dtype=self.args.dtype)
        item_gpu['flat'] = cp.ones(
            [2, self.nflat, self.ncz, self.ni], dtype=self.args.dtype)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.ncz, self.n, self.n], dtype=self.args.dtype))
        # gpu memory for reconstrution
        rec = cp.zeros([2, self.ncz, self.n, self.n], dtype=self.args.dtype)

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)

        # list of procs for parallel writing to hard disk
        write_procs = []

        log.info('Full reconstruction')
        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(self.nzchunk+2):
            utils.printProgressBar(
                k, self.nzchunk+1, self.data_queue.qsize(), length=40)
            if(k > 0 and k < self.nzchunk+1):
                with stream2:  # reconstruction
                    self.recon(rec[(k-1) % 2], item_gpu['data'][(k-1) % 2],
                               item_gpu['dark'][(k-1) % 2], item_gpu['flat'][(k-1) % 2])
            if(k > 1):
                with stream3:  # gpu->cpu copy
                    rec[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < self.nzchunk):
                # copy to pinned memory
                item = self.data_queue.get()
                item_pinned['data'][k % 2, :, :self.lzchunk[k]] = item['data']
                item_pinned['dark'][k % 2, :, :self.lzchunk[k]] = item['dark']
                item_pinned['flat'][k % 2, :, :self.lzchunk[k]] = item['flat']
                with stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])
                    item_gpu['dark'][k % 2].set(item_pinned['dark'][k % 2])
                    item_gpu['flat'][k % 2].set(item_pinned['flat'][k % 2])
            stream3.synchronize()
            if(k > 1):
                # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                rec_pinned0 = rec_pinned[(k-2) % 2, :self.lzchunk[k-2]].copy()
                write_proc = mp.Process(
                    target=self.cl_conf.write_data, args=(rec_pinned0, k-2))
                write_procs.append(write_proc)
                write_proc.start()
            stream1.synchronize()
            stream2.synchronize()

        # wait until reconstructions are written to hard disk
        for proc in write_procs:
            proc.join()

    def recon_try(self):
        """GPU reconstruction of 1 slice from an h5file"""

        data, flat, dark = self.cl_conf.read_data_try()
        shift_array = self.cl_conf.shift_array

        data = cp.ascontiguousarray(cp.array(data))
        dark = cp.ascontiguousarray(cp.array(dark))
        flat = cp.ascontiguousarray(cp.array(flat))

        # preprocessing 1 slice
        data = self.darkflat_correction(data, dark, flat)
        if(self.args.remove_stripe_method == 'fw'):
            data = remove_stripe.remove_stripe_fw(
                data, self.args.fw_sigma, self.args.fw_filter, self.args.fw_level)
        data = self.minus_log(data)
        if(self.args.file_type == 'double_fov'):
            data = self.pad360(data)
        data = cp.ascontiguousarray(data.swapaxes(0, 1))

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.ncz, self.n, self.n], dtype=self.args.dtype))
        # gpu memory for reconstrution
        rec = cp.zeros([2, self.ncz, self.n, self.n], dtype=self.args.dtype)

        # invert shifts for calculations if centeri<ni for double_fov
        mul = 1
        if (self.args.file_type == 'double_fov') and (self.centeri < self.ni//2):
            mul = -1

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)

        write_procs = []
        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(self.nschunk+2):
            utils.printProgressBar(
                k, self.nschunk+1, self.data_queue.qsize(), length=40)
            if(k > 0 and k < self.nschunk+1):
                with stream2:  # reconstruction
                    datat = self.fbp_filter_center(data, cp.array(
                        mul*shift_array[(k-1)*self.ncz:(k-1)*self.ncz+self.lschunk[k-1]]))  # note multiplication by mul
                    self.cl_rec.backprojection(
                        rec[(k-1) % 2], datat, cp.cuda.get_current_stream())
            if(k > 1):
                with stream3:  # gpu->cpu copy
                    rec[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            stream3.synchronize()
            if(k > 1):
                # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                rec_pinned0 = rec_pinned[(k-2) % 2, :self.lschunk[k-2]].copy()
                for kk in range(self.lschunk[k-2]):
                    cid = (self.centeri -
                           shift_array[(k-2)*self.ncz+kk])*2**self.args.binning
                    write_proc = mp.Process(
                        target=self.cl_conf.write_data_try, args=(rec_pinned0[kk], cid))
                    write_procs.append(write_proc)
                    write_proc.start()
            stream1.synchronize()
            stream2.synchronize()

        for proc in write_procs:
            proc.join()

    def find_center(self):
        from ast import literal_eval
        pairs = literal_eval(self.args.rotation_axis_pairs)

        flat, dark = self.cl_conf.read_flat_dark()
        data = self.cl_conf.read_pairs(pairs)
        data = cp.array(data)
        flat = cp.array(flat)
        dark = cp.array(dark)

        data = self.darkflat_correction(data, dark, flat)
        data = self.minus_log(data)
        data = data.get()
        shifts, nmatches = find_rotation.register_shift_sift(
            data[::2], data[1::2, :, ::-1], self.args.rotation_axis_sift_threshold)
        centers = self.n//2-shifts[:, 1]/2+self.cl_conf.stn
        log.info(f'Number of matched features {nmatches}')
        log.info(
            f'Found centers for projection pairs {centers}, mean: {np.mean(centers)}')
        log.info(
            f'Vertical misalignment {shifts[:, 0]}, mean: {np.mean(shifts[:, 0])}')
        return np.mean(centers)
