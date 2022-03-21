from tomocupyfp16_cli import fourierrec
from tomocupyfp16_cli import retrieve_phase, remove_stripe
from tomocupyfp16_cli import find_rotation
from tomocupyfp16_cli import utils
from tomocupyfp16_cli import logging
from cupyx.scipy.fft import rfft, irfft
import cupy as cp
import numpy as np
import numexpr as ne
import dxchange
import threading
import h5py
import os
import signal

pinned_memory_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

log = logging.getLogger(__name__)


class GPURecSteps():
    def __init__(self, args):
        # Set ^C interrupt to abort and deallocate memory on GPU
        signal.signal(signal.SIGINT, utils.signal_handler)
        signal.signal(signal.SIGTSTP, utils.signal_handler)

        with h5py.File(args.file_name) as fid:
            # determine sizes
            ni = fid['/exchange/data'].shape[2]
            ndark = fid['/exchange/data_dark'].shape[0]
            nflat = fid['/exchange/data_white'].shape[0]
            theta = fid['/exchange/theta'][:].astype('float32')/180*np.pi
            if (args.end_row == -1):
                args.end_row = fid['/exchange/data'].shape[1]
            if (args.end_proj == -1):
                args.end_proj = fid['/exchange/data'].shape[0]
        # define chunk size for processing
        ncz = args.nsino_per_chunk
        ncproj = args.nproj_per_chunk
        # take center
        centeri = args.rotation_axis
        if centeri == -1:
            centeri = ni/2
        # update sizes wrt binning
        ni //= 2**args.binning
        centeri /= 2**args.binning

        # change sizes for 360 deg scans with rotation axis at the border
        if(args.file_type == 'double_fov'):
            n = 2*ni
            if(centeri < ni//2):
                # if rotation center is on the left side of the ROI
                center = ni-centeri
            else:
                center = centeri
        else:
            n = ni
            center = centeri

        # blocked views fix
        ids_proj = np.arange(len(theta))[args.start_proj:args.end_proj]
        theta = theta[ids_proj]

        if args.blocked_views:
            st = args.blocked_views_start
            end = args.blocked_views_end
            ids = np.where(((theta) % np.pi < st) +
                           ((theta-st) % np.pi > end-st))[0]
            theta = theta[ids]
            ids_proj = ids_proj[ids]

        nproj = len(theta)
        theta = cp.ascontiguousarray(cp.array(theta))

        # choose reconstruction method
        if args.reconstruction_algorithm == 'lprec':
            self.cl_rec = lprec.LpRec(n, nproj, ncz)
        if args.reconstruction_algorithm == 'fourierrec':
            self.cl_rec = fourierrec.FourierRec(n, nproj, ncz, theta)

        nz = (args.end_row-args.start_row)//2**args.binning

        self.n = n
        self.nz = nz
        self.ncz = ncz
        self.nproj = nproj
        self.ncproj = ncproj
        self.center = center
        self.ni = ni
        self.centeri = centeri
        self.ndark = ndark
        self.nflat = nflat
        self.ids_proj = ids_proj
        self.args = args

    def downsample(self, data):
        """Downsample data"""

        data = data.astype('float32')
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

        dark0 = cp.mean(dark, axis=0).astype('float16')
        flat0 = cp.mean(flat, axis=0).astype('float16')
        data = (data.astype('float16')-dark0)/(flat0-dark0)
        return data
    
    def minus_log(self, data):
        """Taking negative logarithm"""

        data = -cp.log(data)
        data[cp.isnan(data)] = 6.0
        data[cp.isinf(data)] = 0
        return data

    def fbp_filter_center(self, data, sh=0):
        """FBP filtering of projections"""

        ne = 2**int(np.ceil(np.log2(3*self.n//2)))
        t = cp.fft.rfftfreq(ne).astype('float32')
        w = t * (1 - t * 2)**3  # parzen
        w = w*cp.exp(-2*cp.pi*1j*t*(-self.center+sh+self.n/2))  # center fix
                
        data = cp.pad(
            data, ((0, 0), (0, 0), (ne//2-self.n//2, ne//2-self.n//2)), mode='edge')
        self.cl_rec.filter(data,w,cp.cuda.get_current_stream())
        #data = irfft(
            #w*rfft(data, axis=2), axis=2).astype('float16')  # note: filter works with complex64, however, it doesnt take much time
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

    def read_data(self, data, fdata, k, lchunk):
        """Read a chunk of projection with binning"""

        d = fdata[self.args.start_proj+k*lchunk:self.args.start_proj +
                  (k+1)*lchunk, self.args.start_row:self.args.end_row]
        data[k*lchunk:self.args.start_proj+(k+1)*lchunk] = self.downsample(d)

    def read_data_parallel(self, nthreads=8):
        """Read data in parallel chunks (good for SSD disks)"""

        fid = h5py.File(self.args.file_name, 'r')

        # read dark and flat, binning
        dark = fid['exchange/data_dark'][:,
                                         self.args.start_row:self.args.end_row].astype('float16')
        flat = fid['exchange/data_white'][:,
                                          self.args.start_row:self.args.end_row].astype('float16')
        flat = self.downsample(flat)
        dark = self.downsample(dark)

        # parallel read of projections
        data = np.zeros([self.nproj, self.nz, self.ni],
                        dtype=fid['exchange/data'].dtype)
        lchunk = int(np.ceil(self.nproj/nthreads))
        threads = []
        for k in range(nthreads):
            read_thread = threading.Thread(target=self.read_data, args=(
                data, fid['exchange/data'], k, lchunk))
            threads.append(read_thread)
            read_thread.start()
        for thread in threads:
            thread.join()

        return data, dark, flat

    def recon_steps(self):
        """GPU reconstruction by loading a full dataset in memory and processing by steps """

        log.info('Step 1. Reading data.')
        data, dark, flat = self.read_data_parallel()
        
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

        res = np.zeros(data.shape, dtype='float16')

        nchunk = int(np.ceil(self.nz/self.ncz))
        lchunk = np.minimum(
            self.ncz, np.int32(self.nz-np.arange(nchunk)*self.ncz))  # chunk sizes

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.nproj, self.ncz, self.ni], dtype='float16'))
        item_pinned['dark'] = utils.pinned_array(
            np.zeros([2, self.ndark, self.ncz, self.ni], dtype='float16'))
        item_pinned['flat'] = utils.pinned_array(
            np.ones([2, self.nflat, self.ncz, self.ni], dtype='float16'))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.nproj, self.ncz, self.ni], dtype='float16')
        item_gpu['dark'] = cp.zeros(
            [2, self.ndark, self.ncz, self.ni], dtype='float16')
        item_gpu['flat'] = cp.ones(
            [2, self.nflat, self.ncz, self.ni], dtype='float16')

        # pinned memory for res
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.nproj, self.ncz, self.n], dtype='float16'))
        # gpu memory for res
        rec = cp.zeros([2, self.nproj, self.ncz, self.n], dtype='float16')

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

        res = np.zeros(data.shape, dtype='float16')

        nchunk = int(np.ceil(self.nproj/self.ncproj))
        lchunk = np.minimum(
            self.ncproj, np.int32(self.nproj-np.arange(nchunk)*self.ncproj))  # chunk sizes

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.ncproj, self.nz, self.n], dtype='float16'))
        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.ncproj, self.nz, self.n], dtype='float16')

        # pinned memory for processed data
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.ncproj, self.nz, self.n], dtype='float16'))
        # gpu memory for processed data
        rec = cp.zeros([2, self.ncproj, self.nz, self.n], dtype='float16')
        
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
                # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                res[(k-2)*self.ncproj:(k-2)*self.ncproj+lchunk[k-2]
                    ] = rec_pinned[(k-2) % 2, :lchunk[k-2]].copy()
            stream1.synchronize()
            stream2.synchronize()
        return res

    def rec_sino_parallel(self, data):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        nchunk = int(np.ceil(self.nz/self.ncz))
        lchunk = np.minimum(
            self.ncz, np.int32(self.nz-np.arange(nchunk)*self.ncz))  # chunk sizes

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.nproj, self.ncz, self.ni], dtype='float16'))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.nproj, self.ncz, self.ni], dtype='float16')

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.ncz, self.n, self.n], dtype='float16'))
        # gpu memory for reconstrution
        rec = cp.zeros([2, self.ncz, self.n, self.n], dtype='float16')

        # list of threads for parallel writing to hard disk
        write_threads = []

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)
        if(self.args.out_path_name is None):
            fnameout = os.path.dirname(
                self.args.file_name)+'_rec/'+os.path.basename(self.args.file_name)[:-3]+'_rec/recon'
        else:
            fnameout = str(self.args.out_path_name)+'/r'

        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nchunk+2):
            utils.printProgressBar(k, nchunk+1, nchunk-k+1, length=40)
            if(k > 0 and k < nchunk+1):
                with stream2:  # reconstruction
                    self.rec_sino(rec[(k-1) % 2], item_gpu['data'][(k-1) % 2])
            if(k > 1):
                with stream3:  # gpu->cpu copy
                    rec[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < nchunk):
                # copy to pinned memory
                item_pinned['data'][k % 2, :, :lchunk[k]
                                    ] = data[:, k*self.ncz:k*self.ncz+lchunk[k]]
                with stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])
            stream3.synchronize()
            if(k > 1):
                # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                rec_pinned0 = rec_pinned[(k-2) % 2, :lchunk[k-2]].copy()
                write_thread = threading.Thread(target=dxchange.write_tiff_stack,
                                                args=(rec_pinned0,),
                                                kwargs={'fname': fnameout,
                                                        'start':  (k-2)*self.ncz+self.args.start_row//2**self.args.binning,
                                                        'overwrite': True})
                write_threads.append(write_thread)
                write_thread.start()
            stream1.synchronize()
            stream2.synchronize()
        log.info(f'Output: {fnameout}')
        # wait until reconstructions are written to hard disk
        for thread in write_threads:
            thread.join()
