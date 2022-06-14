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

from hashlib import sha256
from tomocupy_cli import fourierrec
from tomocupy_cli import fbp
from tomocupy_cli import fbp_filter
from tomocupy_cli import retrieve_phase, remove_stripe
from tomocupy_cli import find_rotation
from tomocupy_cli import utils
from tomocupy_cli import logging
from tomocupy_cli import confio
import multiprocessing as mp
import threading
#from cupyx.scipy.fft import rfft, irfft
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

        cl_conf = confio.ConfIO(args)

        self.n = cl_conf.n
        self.ncz = cl_conf.ncz
        self.nz = cl_conf.nz
        self.ncproj = cl_conf.ncproj
        self.nproj = cl_conf.nproj
        self.in_dtype = cl_conf.in_dtype
        self.center = cl_conf.center
        self.ni = cl_conf.ni
        self.centeri = cl_conf.centeri
        self.ndark = cl_conf.ndark
        self.nflat = cl_conf.nflat
        self.ids_proj = cl_conf.ids_proj
        self.rh = cl_conf.rh
        self.theta = cl_conf.theta
        self.shift_array = cl_conf.shift_array
        self.args = cl_conf.args
        self.nzchunk = cl_conf.nzchunk
        self.lzchunk = cl_conf.lzchunk
        self.ntchunk = cl_conf.ntchunk
        self.ltchunk = cl_conf.ltchunk
        self.nschunk = cl_conf.nschunk
        self.lschunk = cl_conf.lschunk
        self.nrchunk = cl_conf.nrchunk
        self.lrchunk = cl_conf.lrchunk
        self.cl_conf = cl_conf

        self.cl_filter = fbp_filter.FBPFilter(
            self.n, self.ncproj, self.nz, args.dtype)
        if self.args.reconstruction_algorithm == 'fourierrec':
            self.cl_rec = fourierrec.FourierRec(
                self.n, self.nproj, self.ncz, cp.array(cl_conf.theta), args.dtype)
        # queue for streaming projections
        self.data_queue = mp.Queue()
        # streams for overlapping data transfers with computations
        self.stream1 = cp.cuda.Stream(non_blocking=False)
        self.stream2 = cp.cuda.Stream(non_blocking=False)
        self.stream3 = cp.cuda.Stream(non_blocking=False)

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
        sht = cp.zeros([self.nz, 1], dtype='float32')

        if isinstance(sh, cp.ndarray):
            # try
            data = cp.tile(data, (self.nz//2, 1, 1))
            sht[:len(sh)] = sh.reshape(len(sh), 1)
        else:
            # full
            sht[:] = sh
        w = w*cp.exp(-2*cp.pi*1j*t*(-self.center+sht+self.n/2))  # center fix
        data = cp.pad(
            data, ((0, 0), (0, 0), (ne//2-self.n//2, ne//2-self.n//2)), mode='edge')
        self.cl_filter.filter(data, w, cp.cuda.get_current_stream())
        # data = irfft(
        # w*rfft(data, axis=2), axis=2).astype(self.args.dtype)  # note: filter works with complex64, however, it doesnt take much time
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

    def proc_sino(self, res, data, dark, flat):
        """Processing a sinogram data chunk"""

        # dark flat field correrction
        data = self.darkflat_correction(data, dark, flat)
        # remove stripes
        if(self.args.remove_stripe_method == 'fw'):
            data = remove_stripe.remove_stripe_fw(
                data, self.args.fw_sigma, self.args.fw_filter, self.args.fw_level)
        res[:] = data

    def proc_proj(self, res, data):
        """Processing a projection data chunk"""

        # retrieve phase
        if(self.args.retrieve_phase_method == 'paganin'):
            data = retrieve_phase.paganin_filter(
                data,  self.args.pixel_size*1e-4, self.args.propagation_distance/10, self.args.energy, self.args.retrieve_phase_alpha)
        # minus log
        data = self.minus_log(data)
        # padding for 360 deg recon
        if(self.args.file_type == 'double_fov'):
            data = self.pad360(data)
        data = self.fbp_filter_center(data)

        res[:] = data

    def recon_steps_all(self):
        """GPU reconstruction by loading a full dataset in memory and processing by steps """

        log.info('Step 1. Reading data.')
        data, flat, dark = self.read_data_parallel()

        log.info('Step 2. Processing by chunks in z.')
        data = self.proc_sino_parallel(data, dark, flat)

        log.info('Step 3. Processing by chunks in angles.')
        data = self.proc_proj_parallel(data)

        # Extra block to find centers
        if self.args.rotation_axis_auto == 'auto':
            from ast import literal_eval
            pairs = literal_eval(self.args.rotation_axis_pairs)
            shifts, nmatches = find_rotation.register_shift_sift(
                data[pairs[::2]], data[pairs[1::2], :, ::-1], self.args.rotation_axis_sift_threshold)
            centers = self.n//2-shifts[:, 1]/2
            log.info(f'Number of matched features {nmatches}')
            log.info(
                f'Found centers for projection pairs {centers}, mean: {np.mean(centers)}')
            log.info(
                f'Vertical misalignment {shifts[:, 0]}, mean: {np.mean(shifts[:, 0])}')

            self.center = np.mean(centers)

        if self.args.reconstruction_algorithm == 'fourierrec':
            log.info('Step 4. Reconstruction by chunks in z.')
            data = np.ascontiguousarray(data.swapaxes(0, 1))
            self.recon_sino_parallel(data)
        if self.args.reconstruction_algorithm == 'fbp':
            log.info('Step 4. Reconstruction by chunks in z and angles.')
            self.recon_sino_proj_parallel(data)

    def recon_steps_try(self):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        log.info('Step 1. Reading data.')
        data, flat, dark = self.read_data_parallel()

        log.info('Step 2. Processing by chunks in z.')
        data = self.proc_sino_parallel(data, dark, flat)

        # exit()
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
        data = np.zeros([self.nproj, self.nz, self.ni], dtype=self.args.dtype)
        lchunk = int(np.ceil(self.nproj/nproc))
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

        res = np.zeros(data.shape, dtype=self.args.dtype)

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

        # pinned memory for res
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.nproj, self.ncz, self.ni], dtype=self.args.dtype))
        # gpu memory for res
        rec = cp.zeros([2, self.nproj, self.ncz, self.ni],
                       dtype=self.args.dtype)

        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(self.nzchunk+2):
            utils.printProgressBar(
                k, self.nzchunk+1, self.nzchunk-k+1, length=40)

            if(k > 0 and k < self.nzchunk+1):
                with self.stream2:  # reconstruction
                    self.proc_sino(rec[(k-1) % 2], item_gpu['data'][(k-1) % 2],
                                   item_gpu['dark'][(k-1) % 2], item_gpu['flat'][(k-1) % 2])
            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    rec[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < self.nzchunk):
                # copy to pinned memory

                item_pinned['data'][k % 2, :, :self.lzchunk[k]
                                    ] = data[:, k*self.ncz:k*self.ncz+self.lzchunk[k]]
                item_pinned['dark'][k % 2, :, :self.lzchunk[k]
                                    ] = dark[:, k*self.ncz:k*self.ncz+self.lzchunk[k]]
                item_pinned['flat'][k % 2, :, :self.lzchunk[k]
                                    ] = flat[:, k*self.ncz:k*self.ncz+self.lzchunk[k]]
                with self.stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])
                    item_gpu['dark'][k % 2].set(item_pinned['dark'][k % 2])
                    item_gpu['flat'][k % 2].set(item_pinned['flat'][k % 2])
            self.stream3.synchronize()
            if(k > 1):
                res[:, (k-2)*self.ncz:(k-2)*self.ncz+self.lzchunk[k-2]
                    ] = rec_pinned[(k-2) % 2, :, :self.lzchunk[k-2]].copy()
            self.stream1.synchronize()
            self.stream2.synchronize()
        return res

    def proc_proj_parallel(self, data):

        res = np.zeros(data.shape, dtype=self.args.dtype)

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.ncproj, self.nz, self.ni], dtype=self.args.dtype))
        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.ncproj, self.nz, self.ni], dtype=self.args.dtype)

        # pinned memory for processed data
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.ncproj, self.nz, self.ni], dtype=self.args.dtype))
        # gpu memory for processed data
        rec = cp.zeros([2, self.ncproj, self.nz, self.ni],
                       dtype=self.args.dtype)

        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(self.ntchunk+2):

            utils.printProgressBar(
                k, self.ntchunk+1, self.ntchunk-k+1, length=40)
            if(k > 0 and k < self.ntchunk+1):
                with self.stream2:  # reconstruction
                    self.proc_proj(rec[(k-1) % 2], item_gpu['data'][(k-1) % 2])

            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    rec[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < self.ntchunk):
                # copy to pinned memory
                item_pinned['data'][k % 2, :self.ltchunk[k]
                                    ] = data[self.ncproj*k:self.ncproj*k+self.ltchunk[k]]
                with self.stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])

            self.stream3.synchronize()
            if(k > 1):
                # add a new proc for writing to hard disk (after gpu->cpu copy is done)
                res[(k-2)*self.ncproj:(k-2)*self.ncproj+self.ltchunk[k-2]
                    ] = rec_pinned[(k-2) % 2, :self.ltchunk[k-2]].copy()
            self.stream1.synchronize()
            self.stream2.synchronize()
        return res

    def recon_sino_parallel(self, data):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.ncz, self.nproj, self.ni], dtype=self.args.dtype))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.ncz, self.nproj, self.ni], dtype=self.args.dtype)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.ncz, self.n, self.n], dtype=self.args.dtype))
        # gpu memory for reconstrution
        rec = cp.zeros([2, self.ncz, self.n, self.n], dtype=self.args.dtype)

        # list of procs for parallel writing to hard disk
        write_procs = []

        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(self.nzchunk+2):
            utils.printProgressBar(
                k, self.nzchunk+1, self.nzchunk-k+1, length=40)
            if(k > 0 and k < self.nzchunk+1):
                with self.stream2:  # reconstruction
                    self.cl_rec.backprojection(
                        rec[(k-1) % 2], item_gpu['data'][(k-1) % 2], cp.cuda.get_current_stream())
            if(k > 1):
                with self.stream3:  # gpu->cpu copy
                    rec[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < self.nzchunk):
                # copy to pinned memory
                item_pinned['data'][k % 2, :self.lzchunk[k]
                                    ] = data[k*self.ncz:k*self.ncz+self.lzchunk[k]]
                with self.stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])
            self.stream3.synchronize()
            if(k > 1):
                # add a new proc for writing to hard disk (after gpu->cpu copy is done)
                rec_pinned0 = rec_pinned[(k-2) % 2, :self.lzchunk[k-2]].copy()
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

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.ncproj, self.nz, self.ni], dtype='float32'))
        item_pinned['theta'] = utils.pinned_array(
            np.zeros([2, self.ncproj], dtype='float32'))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.ncproj, self.nz, self.ni], dtype='float32')
        item_gpu['theta'] = cp.zeros(
            [2, self.ncproj], dtype='float32')

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.ncz, self.n, self.n], dtype='float32'))
        # gpu memory for reconstrution
        rec = cp.zeros([2, self.ncz, self.n, self.n], dtype='float32')

        # list of procs for parallel writing to hard disk
        write_procs = []

        # Conveyor for data cpu-gpu copy and reconstruction
        for kr in range(self.nrchunk+2):
            utils.printProgressBar(
                kr, self.nrchunk+1, self.nrchunk-kr+1, length=40)
            rec[(kr-1) % 2][:] = 0
            for kt in range(self.ntchunk+2):
                if (kr > 0 and kr < self.nrchunk+1 and kt > 0 and kt < self.ntchunk+1):
                    with self.stream2:  # reconstruction
                        fbp.adj(rec[(kr-1) % 2],
                                item_gpu['data'][(kt-1) % 2],
                                item_gpu['theta'][(kt-1) % 2],
                                self.args.lamino_angle, (kr-1)*self.ncz)

                if (kr > 1 and kt == 0):
                    with self.stream3:  # gpu->cpu copy
                        rec[(kr-2) % 2, :] = rec[(kr-2) % 2, :, ::-1]
                        rec[(kr-2) % 2].get(out=rec_pinned[(kr-2) % 2])
                if(kt < self.ntchunk):
                    # copy to pinned memory
                    item_pinned['data'][kt % 2][:self.ltchunk[kt]
                                                ] = data[kt*self.ncproj:kt*self.ncproj+self.ltchunk[kt]]
                    item_pinned['theta'][kt % 2][:self.ltchunk[kt]
                                                 ] = self.theta[kt*self.ncproj:kt*self.ncproj+self.ltchunk[kt]]
                    item_pinned['data'][kt % 2][self.ltchunk[kt]:] = 0
                    with self.stream1:  # cpu->gpu copy
                        item_gpu['data'][kt % 2].set(
                            item_pinned['data'][kt % 2])
                        item_gpu['theta'][kt % 2].set(
                            item_pinned['theta'][kt % 2])
                self.stream3.synchronize()
                if (kr > 1 and kt == 0):
                    # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                    rec_pinned0 = rec_pinned[(kr-2) %
                                             2, :self.lrchunk[kr-2]].copy()
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
        # recon slice
        rslice = int(self.args.nsino*self.rh)
        log.info(
            f'Reconstruction of the slice {rslice} for centers {self.center+self.shift_array}')

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.ncproj, self.nz, self.ni], dtype='float32'))
        item_pinned['theta'] = utils.pinned_array(
            np.zeros([2, self.ncproj], dtype='float32'))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.ncproj, self.nz, self.ni], dtype='float32')
        item_gpu['theta'] = cp.zeros(
            [2, self.ncproj], dtype='float32')

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.ncz, self.n, self.n], dtype='float32'))
        # gpu memory for reconstrution
        rec = cp.zeros([2, self.ncz, self.n, self.n], dtype='float32')

        # invert shifts for calculations if centeri<ni for double_fov
        mul = 1
        if (self.args.file_type == 'double_fov') and (self.centeri < self.ni//2):
            mul = -1
        sh = mul*cp.array(self.shift_array)

        write_procs = []
        # Conveyor for data cpu-gpu copy and reconstruction
        for ks in range(self.nschunk+2):
            rec[(ks-1) % 2][:] = 0

            for kt in range(self.ntchunk+2):
                if (ks > 0 and ks < self.nschunk+1 and kt > 0 and kt < self.ntchunk+1):
                    with self.stream2:  # reconstruction
                        sh0 = sh[(ks-1)*self.ncz:(ks-1) *
                                 self.ncz+self.lschunk[ks-1]]
                        fbp.adj_try(rec[(ks-1) % 2],
                                    item_gpu['data'][(kt-1) % 2],
                                    item_gpu['theta'][(kt-1) % 2],
                                    self.args.lamino_angle, rslice, sh0)
                if (ks > 1 and kt == 0):
                    with self.stream3:  # gpu->cpu copy
                        rec[(ks-2) % 2] = rec[(ks-2) % 2, :, ::-1]
                        rec[(ks-2) % 2].get(out=rec_pinned[(ks-2) % 2])
                if(kt < self.ntchunk):
                    # copy to pinned memory
                    item_pinned['data'][kt % 2][:self.ltchunk[kt]
                                                ] = data[kt*self.ncproj:kt*self.ncproj+self.ltchunk[kt]]
                    item_pinned['theta'][kt % 2][:self.ltchunk[kt]
                                                 ] = self.theta[kt*self.ncproj:kt*self.ncproj+self.ltchunk[kt]]
                    item_pinned['data'][kt % 2][self.ltchunk[kt]:] = 0
                    with self.stream1:  # cpu->gpu copy
                        item_gpu['data'][kt % 2].set(
                            item_pinned['data'][kt % 2])
                        item_gpu['theta'][kt % 2].set(
                            item_pinned['theta'][kt % 2])
                self.stream3.synchronize()
                if (ks > 1 and kt == 0):
                    rec_pinned0 = rec_pinned[(ks-2) %
                                             2, :self.lschunk[ks-2]].copy()
                    # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                    for kk in range(self.lschunk[ks-2]):
                        cid = (self.centeri -
                               self.shift_array[(ks-2)*self.ncz+kk])*2**self.args.binning
                        write_proc = mp.Process(
                            target=self.cl_conf.write_data_try, args=(rec_pinned0[kk], cid))
                        write_procs.append(write_proc)
                        write_proc.start()
                self.stream1.synchronize()
                self.stream2.synchronize()

    def recon_try_lamino_sino_proj_parallel(self, data):
        # recon slice
        rslice = int(self.args.nsino*self.rh)
        log.info(
            f'Reconstruction of the slice {rslice} for lamino angles {self.args.lamino_angle+self.shift_array}')

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.ncproj, self.nz, self.ni], dtype='float32'))
        item_pinned['theta'] = utils.pinned_array(
            np.zeros([2, self.ncproj], dtype='float32'))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.ncproj, self.nz, self.ni], dtype='float32')
        item_gpu['theta'] = cp.zeros(
            [2, self.ncproj], dtype='float32')

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.ncz, self.n, self.n], dtype='float32'))
        # gpu memory for reconstrution
        rec = cp.zeros([2, self.ncz, self.n, self.n], dtype='float32')

        sh = cp.array(self.shift_array)

        write_procs = []
        # Conveyor for data cpu-gpu copy and reconstruction
        for ks in range(self.nschunk+2):
            rec[(ks-1) % 2][:] = 0

            for kt in range(self.ntchunk+2):
                if (ks > 0 and ks < self.nschunk+1 and kt > 0 and kt < self.ntchunk+1):
                    with self.stream2:  # reconstruction
                        sh0 = sh[(ks-1)*self.ncz:(ks-1) *
                                 self.ncz+self.lschunk[ks-1]]
                        fbp.adj_try_lamino(rec[(ks-1) % 2],
                                           item_gpu['data'][(kt-1) % 2],
                                           item_gpu['theta'][(kt-1) % 2],
                                           self.args.lamino_angle, rslice, sh0)
                if (ks > 1 and kt == 0):
                    with self.stream3:  # gpu->cpu copy
                        rec[(ks-2) % 2] = rec[(ks-2) % 2, :, ::-1]
                        rec[(ks-2) % 2].get(out=rec_pinned[(ks-2) % 2])
                if(kt < self.ntchunk):
                    # copy to pinned memory
                    item_pinned['data'][kt % 2][:self.ltchunk[kt]
                                                ] = data[kt*self.ncproj:kt*self.ncproj+self.ltchunk[kt]]
                    item_pinned['theta'][kt % 2][:self.ltchunk[kt]
                                                 ] = self.theta[kt*self.ncproj:kt*self.ncproj+self.ltchunk[kt]]
                    item_pinned['data'][kt % 2][self.ltchunk[kt]:] = 0
                    with self.stream1:  # cpu->gpu copy
                        item_gpu['data'][kt % 2].set(
                            item_pinned['data'][kt % 2])
                        item_gpu['theta'][kt % 2].set(
                            item_pinned['theta'][kt % 2])
                self.stream3.synchronize()
                if (ks > 1 and kt == 0):
                    rec_pinned0 = rec_pinned[(ks-2) %
                                             2, :self.lschunk[ks-2]].copy()
                    # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                    for kk in range(self.lschunk[ks-2]):
                        cid = self.args.lamino_angle + \
                            self.shift_array[(ks-2)*self.ncz+kk]
                        write_proc = mp.Process(
                            target=self.cl_conf.write_data_try, args=(rec_pinned0[kk], cid))
                        write_procs.append(write_proc)
                        write_proc.start()
                self.stream1.synchronize()
                self.stream2.synchronize()
