from tomocupy_cli import lprec, fourierrec
from tomocupy_cli import remove_stripe
from tomocupy_cli import utils
from tomocupy_cli import logging
from cupyx.scipy.fft import rfft, irfft
import cupy as cp
import numpy as np
import numexpr as ne
import dxchange
import threading
import queue
import h5py
import os
import signal

pinned_memory_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

log = logging.getLogger(__name__)

class GPURec():
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
            if (args.end_row==-1):
                args.end_row = fid['/exchange/data'].shape[1]
            if (args.end_proj==-1):
                args.end_proj = fid['/exchange/data'].shape[0]
        # define chunk size for processing
        nz = args.nsino_per_chunk
        if(args.reconstruction_type == 'try'):
            nz = 2
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
        theta = cp.array(theta)

        # choose reconstruction method
        if args.reconstruction_algorithm == 'lprec':
            self.cl_rec = lprec.LpRec(n, nproj, nz)
        if args.reconstruction_algorithm == 'fourierrec':
            self.cl_rec = fourierrec.FourierRec(n, nproj, nz, theta)

        
        self.n = n
        self.nz = nz
        self.nproj = nproj
        self.center = center
        self.ni = ni
        self.centeri = centeri
        self.ndark = ndark
        self.nflat = nflat
        self.ids_proj = ids_proj        
        self.args = args

        # queue for streaming projections
        self.data_queue = queue.Queue()

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

        dark0 = cp.mean(dark, axis=0).astype('float32')
        flat0 = cp.mean(flat, axis=0).astype('float32')
        data = (data.astype('float32')-dark0)/(flat0-dark0)
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
        t = cp.fft.rfftfreq(ne).astype('float32')
        w = t * (1 - t * 2)**3  # parzen
        w = w*cp.exp(-2*cp.pi*1j*t*(-self.center+sh+self.n/2))  # center fix
        data = cp.pad(
            data, ((0, 0), (0, 0), (ne//2-self.n//2, ne//2-self.n//2)), mode='edge')

        data = irfft(
            w*rfft(data, axis=2), axis=2).astype('float32')  # note: filter works with complex64, however, it doesnt take much time
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
        # remove stripes
        if(self.args.remove_stripe_method == 'fw'):
            data = remove_stripe.remove_stripe_fw(data, self.args.fw_sigma, self.args.fw_filter, self.args.fw_level)
                
        # minus log
        data = self.minus_log(data)
        
        # padding for 360 deg recon
        if(self.args.file_type == 'double_fov'):
            data = self.pad360(data)
        # fbp filter and compensatio for the center
        data = self.fbp_filter_center(data)
        # reshape to sinograms
        data = cp.ascontiguousarray(data.swapaxes(0, 1))
        # backprojection
        self.cl_rec.backprojection(obj, data, cp.cuda.get_current_stream())

    def read_data(self, data, dark, flat, nchunk, lchunk):
        """Reading data from hard disk and putting it to a queue"""

        st_row = self.args.start_row
        
        for k in range(nchunk):
            item = {}
            st = st_row+k*self.nz*2**self.args.binning
            end = st_row+(k*self.nz+lchunk[k])*2**self.args.binning
            item['data'] = self.downsample(data[self.ids_proj,  st:end])
            item['flat'] = self.downsample(flat[:,  st:end])
            item['dark'] = self.downsample(dark[:,  st:end])

            self.data_queue.put(item)

    def recon_all(self):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        # take links to datasets
        fid = h5py.File(self.args.file_name, 'r')

        # init references to data, no reading at this point
        data = fid['exchange/data']
        dark = fid['exchange/data_dark']
        flat = fid['exchange/data_white']
        
        # calculate chunks
        if self.args.end_row==-1:            
            nrow = data.shape[1]-self.args.start_row
        else:
            nrow = self.args.end_row-self.args.start_row
        nchunk = int(np.ceil(nrow/2**self.args.binning/self.nz))
        lchunk = np.minimum(
            self.nz, np.int32(nrow/2**self.args.binning-np.arange(nchunk)*self.nz))  # chunk sizes
        # start reading data to a queue
        read_thread = threading.Thread(
            target=self.read_data, args=(data, dark, flat, nchunk, lchunk))
        read_thread.start()

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.nproj, self.nz, self.ni], dtype='float32'))
        item_pinned['dark'] = utils.pinned_array(
            np.zeros([2, self.ndark, self.nz, self.ni], dtype='float32'))
        item_pinned['flat'] = utils.pinned_array(
            np.ones([2, self.nflat, self.nz, self.ni], dtype='float32'))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.nproj, self.nz, self.ni], dtype='float32')
        item_gpu['dark'] = cp.zeros(
            [2, self.ndark, self.nz, self.ni], dtype='float32')
        item_gpu['flat'] = cp.ones(
            [2, self.nflat, self.nz, self.ni], dtype='float32')

        # pinned memory for reconstrution
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.nz, self.n, self.n], dtype='float32'))
        # gpu memory for reconstrution
        rec = cp.zeros([2, self.nz, self.n, self.n], dtype='float32')

        # list of threads for parallel writing to hard disk
        write_threads = []

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)
        if(self.args.out_path_name is None):
            fnameout = os.path.dirname(
                self.args.file_name)+'_recgpu/'+os.path.basename(self.args.file_name)[:-3]+'_rec/r'
        else:
            fnameout = str(self.args.out_path_name)+'/r'

        log.info('Full reconstruction')
        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nchunk+2):
            utils.printProgressBar(k, nchunk+1, self.data_queue.qsize(), length=40)
            if(k > 0 and k < nchunk+1):
                with stream2:  # reconstruction
                    self.recon(rec[(k-1) % 2], item_gpu['data'][(k-1) % 2],
                               item_gpu['dark'][(k-1) % 2], item_gpu['flat'][(k-1) % 2])
            if(k > 1):
                with stream3:  # gpu->cpu copy
                    rec[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < nchunk):
                # copy to pinned memory
                item = self.data_queue.get()
                item_pinned['data'][k % 2, :, :lchunk[k]] = item['data']
                item_pinned['dark'][k % 2, :, :lchunk[k]] = item['dark']
                item_pinned['flat'][k % 2, :, :lchunk[k]] = item['flat']
                with stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])
                    item_gpu['dark'][k % 2].set(item_pinned['dark'][k % 2])
                    item_gpu['flat'][k % 2].set(item_pinned['flat'][k % 2])
            stream3.synchronize()
            if(k > 1):
                # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                rec_pinned0 = rec_pinned[(k-2) % 2, :lchunk[k-2], ::-1].copy()
                
                write_thread = threading.Thread(target=dxchange.write_tiff_stack,
                                                args=(rec_pinned0,),
                                                kwargs={'fname': fnameout,
                                                        'start':  (k-2)*self.nz+self.args.start_row//2**self.args.binning,
                                                        'overwrite': True})
                write_threads.append(write_thread)
                write_thread.start()
            stream1.synchronize()
            stream2.synchronize()
        log.info(f'Output: {fnameout}')
        # wait until reconstructions are written to hard disk
        for thread in write_threads:
            thread.join()

    def recon_try(self, obj, data, dark, flat, shift_array):
        """Full reconstruction pipeline for 1 slice with different centers"""

        data = self.darkflat_correction(data, dark, flat)

        if(self.args.remove_stripe_method == 'fw'):
            data = remove_stripe.remove_stripe_fw(data, self.args.fw_sigma, self.args.fw_filter, self.args.fw_level)
        data = self.minus_log(data)
        if(self.args.file_type == 'double_fov'):
            data = self.pad360(data)
        rec_cpu_list = []
        data0 = data.copy()
        for k in range(len(shift_array)):
            utils.printProgressBar(k, len(shift_array)-1, 0, length=40)
            data = self.fbp_filter_center(data0, shift_array[k])
            data = cp.ascontiguousarray(data.swapaxes(0, 1))
            self.cl_rec.backprojection(obj, data, cp.cuda.get_current_stream())
            rec_cpu_list.append(obj[0].get())
        return rec_cpu_list

    def recon_all_try(self):
        """GPU reconstruction of 1 slice from an h5file"""
        
        
        # take links to datasets
        fid = h5py.File(self.args.file_name, 'r')
        data = fid['exchange/data']
        dark = fid['exchange/data_dark']
        flat = fid['exchange/data_white']
        idslice = int(self.args.nsino*(data.shape[1]-1))
        log.info(f'Try rotation center reconstruction for slice {idslice}')
        data = data[self.args.start_proj:self.args.end_proj, idslice:idslice+2*2**self.args.binning]
        dark = dark[:, idslice:idslice+2*2**self.args.binning].astype('float32')
        flat = flat[:, idslice:idslice+2*2**self.args.binning].astype('float32')
        data = self.downsample(data)
        flat = self.downsample(flat)
        dark = self.downsample(dark)
        data = cp.ascontiguousarray(cp.array(data))
        dark = cp.ascontiguousarray(cp.array(dark))
        flat = cp.ascontiguousarray(cp.array(flat))

        rec = cp.zeros([self.nz, self.n, self.n], dtype='float32')
        shift_array = np.arange(-self.args.center_search_width,
                                self.args.center_search_width, self.args.center_search_step).astype('float32')

        with cp.cuda.Stream(non_blocking=False):
            rec_cpu_list = self.recon_try(rec, data, dark, flat, shift_array)        
        fnameout = os.path.dirname(
            self.args.file_name)+'_recgpu/try_center/'+os.path.basename(self.args.file_name)[:-3]+'/r_'
        log.info(f'Output: {fnameout}')
        write_threads = []
        # avoid simultaneous directory creation
        dxchange.write_tiff(
            rec_cpu_list[0], f'{fnameout}{((self.centeri-shift_array[0])*2**self.args.binning):08.2f}', overwrite=True)
        for k in range(1, len(shift_array)):
            write_thread = threading.Thread(target=dxchange.write_tiff,
                                            args=(rec_cpu_list[k],),
                                            kwargs={'fname': f'{fnameout}{((self.centeri-shift_array[k])*2**self.args.binning):08.2f}',
                                                    'overwrite': True})
            write_threads.append(write_thread)
            write_thread.start()
        for thread in write_threads:
            thread.join()
