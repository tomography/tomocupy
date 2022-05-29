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

        if args.reconstruction_type == 'try':
            log.warning(
                'Try is not supported for the steps reconstruction, switching to full')
            args.reconstruction_type = 'full'

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
        self.nchunk = cl_conf.nchunk
        self.lchunk = cl_conf.lchunk
        self.args = cl_conf.args

        if args.reconstruction_algorithm == 'fourierrec':
            theta = cp.array(cl_conf.theta)
            self.cl_rec = fourierrec.FourierRec(
                self.n, self.nproj, self.ncz, theta, args.dtype)

        # queue for streaming projections
        self.data_queue = mp.Queue()
        self.cl_conf = cl_conf
#        print(vars(self))

    def downsample(self, data):
        """Downsample data"""

        data = data.astype(self.args.dtype)
        for j in range(self.args.binning):
            x = data[:, :, ::2]
            y = data[:, :, 1::2]
            data = ne.evaluate('x + y')  # should use multithreading
        for k in range(self.args.binning):
            x = data[:, ::2]
            y = data[:, 1::2]
            data = ne.evaluate('x + y')
        return data

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
        self.cl_rec.filter(data, w, cp.cuda.get_current_stream())
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
        res[:] = data

    def rec_sino(self, res, data):
        """Reconstruction of a sinogram data chunk"""

        # padding for 360 deg recon
        if(self.args.file_type == 'double_fov'):
            data = self.pad360(data)
        # fbp filter and compensatio for the center
        data = self.fbp_filter_center(data)
        # reshape to sinograms
        data = cp.ascontiguousarray(data.swapaxes(0, 1))
        # backprojection
        self.cl_rec.backprojection(res, data, cp.cuda.get_current_stream())

    def read_data_parallel(self, nproc=8):
        """Readin data in parallel (good for ssd disks)"""

        flat, dark = self.cl_conf.read_flat_dark()
        # parallel read of projections
        data = np.zeros([self.nproj, self.nz, self.ni], dtype=self.in_dtype)
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

    def recon_steps(self):
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
            shifts = find_rotation.register_shift_sift(
                data[pairs[::2]], data[pairs[1::2], :, ::-1])
            centers = self.n//2-shifts[:, 1]/2
            log.info(
                f'Found centers for projection pairs {centers}, mean: {np.mean(centers)}')
            self.center = np.mean(centers)

        log.info('Step 4. Reconstruction by chunks in z.')
        self.rec_sino_parallel(data)


############################################### Parallel conveyor execution #############################################

    def proc_sino_parallel(self, data, dark, flat):

        res = np.zeros(data.shape, dtype=self.args.dtype)

        nchunk = int(np.ceil(self.nz/self.ncz))
        lchunk = np.minimum(
            self.ncz, np.int32(self.nz-np.arange(nchunk)*self.ncz))  # chunk sizes

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

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)

        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nchunk+2):
            utils.printProgressBar(k, nchunk+1, nchunk-k+1, length=40)

            if(k > 0 and k < nchunk+1):
                with stream2:  # reconstruction
                    self.proc_sino(rec[(k-1) % 2], item_gpu['data'][(k-1) % 2],
                                   item_gpu['dark'][(k-1) % 2], item_gpu['flat'][(k-1) % 2])
            if(k > 1):
                with stream3:  # gpu->cpu copy
                    rec[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < nchunk):
                # copy to pinned memory
                item_pinned['data'][k % 2, :, :lchunk[k]
                                    ] = data[:, k*self.ncz:k*self.ncz+lchunk[k]]
                item_pinned['dark'][k % 2, :, :lchunk[k]
                                    ] = dark[:, k*self.ncz:k*self.ncz+lchunk[k]]
                item_pinned['flat'][k % 2, :, :lchunk[k]
                                    ] = flat[:, k*self.ncz:k*self.ncz+lchunk[k]]
                with stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])
                    item_gpu['dark'][k % 2].set(item_pinned['dark'][k % 2])
                    item_gpu['flat'][k % 2].set(item_pinned['flat'][k % 2])
            stream3.synchronize()
            if(k > 1):
                res[:, (k-2)*self.ncz:(k-2)*self.ncz+lchunk[k-2]
                    ] = rec_pinned[(k-2) % 2, :, :lchunk[k-2]].copy()
            stream1.synchronize()
            stream2.synchronize()
        return res

    def proc_proj_parallel(self, data):

        res = np.zeros(data.shape, dtype=self.args.dtype)

        nchunk = int(np.ceil(self.nproj/self.ncproj))
        lchunk = np.minimum(
            self.ncproj, np.int32(self.nproj-np.arange(nchunk)*self.ncproj))  # chunk sizes

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

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)
        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nchunk+2):
            utils.printProgressBar(k, nchunk+1, nchunk-k+1, length=40)
            if(k > 0 and k < nchunk+1):
                with stream2:  # reconstruction
                    self.proc_proj(rec[(k-1) % 2], item_gpu['data'][(k-1) % 2])

            if(k > 1):
                with stream3:  # gpu->cpu copy
                    rec[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < nchunk):
                # copy to pinned memory
                item_pinned['data'][k % 2, :lchunk[k]
                                    ] = data[self.ncproj*k:self.ncproj*k+lchunk[k]]
                with stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])

            stream3.synchronize()
            if(k > 1):
                # add a new proc for writing to hard disk (after gpu->cpu copy is done)
                res[(k-2)*self.ncproj:(k-2)*self.ncproj+lchunk[k-2]
                    ] = rec_pinned[(k-2) % 2, :lchunk[k-2]].copy()
            stream1.synchronize()
            stream2.synchronize()
        return res

    def rec_sino_parallel(self, data):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.nproj, self.ncz, self.ni], dtype=self.args.dtype))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.nproj, self.ncz, self.ni], dtype=self.args.dtype)

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.ncz, self.n, self.n], dtype=self.args.dtype))
        # gpu memory for reconstrution
        rec = cp.zeros([2, self.ncz, self.n, self.n], dtype=self.args.dtype)

        # list of procs for parallel writing to hard disk
        write_procs = []

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)

        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(self.nchunk+2):
            utils.printProgressBar(
                k, self.nchunk+1, self.nchunk-k+1, length=40)
            if(k > 0 and k < self.nchunk+1):
                with stream2:  # reconstruction
                    self.rec_sino(rec[(k-1) % 2], item_gpu['data'][(k-1) % 2])
            if(k > 1):
                with stream3:  # gpu->cpu copy
                    rec[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < self.nchunk):
                # copy to pinned memory
                item_pinned['data'][k % 2, :, :self.lchunk[k]
                                    ] = data[:, k*self.ncz:k*self.ncz+self.lchunk[k]]
                with stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])
            stream3.synchronize()
            if(k > 1):
                # add a new proc for writing to hard disk (after gpu->cpu copy is done)
                rec_pinned0 = rec_pinned[(k-2) % 2, :self.lchunk[k-2]].copy()
                write_proc = mp.Process(
                    target=self.cl_conf.write_data, args=(rec_pinned0, k-2))
                write_procs.append(write_proc)
                write_proc.start()

            stream1.synchronize()
            stream2.synchronize()
        # wait until reconstructions are written to hard disk
        for proc in write_procs:
            proc.join()
