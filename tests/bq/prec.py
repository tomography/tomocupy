from tomocupy import lprec, fourierrec
from tomocupy import utils
from tomocupy import remove_stripe
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

class GPUPRec():
    def __init__(self, args):
        # Set ^C interrupt to abort and deallocate memory on GPU
        signal.signal(signal.SIGINT, utils.signal_handler)
        signal.signal(signal.SIGTSTP, utils.signal_handler)

        ni = dxchange.read_tiff(args.file_name).shape[1]
        # define chunk size for processing
        nz = args.nsino_per_chunk
        if(args.reconstruction_type == 'try'):
            nz = 2
        # take center
        centeri = args.rotation_axis
        if centeri == -1:
            centeri = ni/2
        
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
        
        theta = np.load(os.path.dirname(args.file_name)+'/angles.npy')                
        nproj = len(theta)
        theta = cp.ascontiguousarray(cp.array(theta))

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
        self.args = args

        # queue for streaming projections
        self.data_queue = queue.Queue()

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

    def recon(self, obj, data):
        """Full reconstruction pipeline for a data chunk"""

        # remove stripes 
        if(self.args.remove_stripe_method == 'fw'):
            data = remove_stripe.remove_stripe_fw(data,order_proj=False)        
        # padding for 360 deg recon
        if(self.args.file_type == 'double_fov'):
            data = self.pad360(data)
        # fbp filter and compensatio for the center
        data = self.fbp_filter_center(data)
        # backprojection
        self.cl_rec.backprojection(obj, data, cp.cuda.get_current_stream())

    def read_data(self, fname, nchunk, lchunk):
        """Reading data from hard disk and putting it to a queue"""

        for k in range(nchunk):
            item = {}
            st = k*self.nz
            end = (k*self.nz+lchunk[k])

            item['data'] = dxchange.read_tiff_stack(fname,ind = range(self.nproj), slc=((st,end,1),(0,self.ni,1))).swapaxes(0,1)#switch to sinogram
            self.data_queue.put(item)

    def recon_all(self):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        nrow = dxchange.read_tiff(self.args.file_name).shape[0]
        nchunk = int(np.ceil(nrow/self.nz))
        lchunk = np.minimum(
            self.nz, np.int32(nrow-np.arange(nchunk)*self.nz))  # chunk sizes
        # start reading data to a queue
        read_thread = threading.Thread(
            target=self.read_data, args=(self.args.file_name, nchunk, lchunk))
        read_thread.start()

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.nz, self.nproj, self.ni], dtype='float32'))
        
        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.nz, self.nproj, self.ni], dtype='float32')
        
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
            file_name_tmp = os.path.dirname(self.args.file_name)
            fnameout = os.path.dirname(file_name_tmp)+'/'+os.path.basename(file_name_tmp)+'_rec/r'
        else:
            fnameout = str(self.args.out_path_name)+'/r'

        print('Reconstruction Full')
        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nchunk+2):
            utils.printProgressBar(k, nchunk+1, self.data_queue.qsize(), length=40)
            if(k > 0 and k < nchunk+1):
                with stream2:  # reconstruction
                    self.recon(rec[(k-1) % 2], item_gpu['data'][(k-1) % 2])
            if(k > 1):
                with stream3:  # gpu->cpu copy
                    rec[(k-2) % 2].get(out=rec_pinned[(k-2) % 2])
            if(k < nchunk):
                # copy to pinned memory
                item = self.data_queue.get()
                item_pinned['data'][k % 2, :lchunk[k]] = item['data']
                with stream1:  # cpu->gpu copy
                    item_gpu['data'][k % 2].set(item_pinned['data'][k % 2])                
            stream3.synchronize()
            if(k > 1):
                # add a new thread for writing to hard disk (after gpu->cpu copy is done)
                rec_pinned0 = rec_pinned[(k-2) % 2, :lchunk[k-2], ::-1].copy()
                                
                write_thread = threading.Thread(target=dxchange.write_tiff_stack,
                                                args=(rec_pinned0,),
                                                kwargs={'fname': fnameout,
                                                        'start':  (k-2)*self.nz,
                                                        'overwrite': True})
                write_threads.append(write_thread)
                write_thread.start()
            stream1.synchronize()
            stream2.synchronize()
        print('wait until all tiffs are saved')
        print(f'{fnameout}')
        # wait until reconstructions are written to hard disk
        for thread in write_threads:
            thread.join()

    def recon_try(self, obj, data, shift_array):
        """Full reconstruction pipeline for 1 slice with different centers"""

        if(self.args.file_type == 'double_fov'):
            data = self.pad360(data)
        rec_cpu_list = []
        data0 = data.copy()
        for k in range(len(shift_array)):
            utils.printProgressBar(k, len(shift_array)-1, 0, length=40)
            data = self.fbp_filter_center(data0, shift_array[k])
            self.cl_rec.backprojection(obj, data, cp.cuda.get_current_stream())
            rec_cpu_list.append(obj[0].get())
        return rec_cpu_list

    def recon_all_try(self):
        """GPU reconstruction of 1 slice from an h5file"""
        
        
        nrow = dxchange.read_tiff(self.args.file_name).shape[0]        
        idslice = int(self.args.nsino*(nrow-1))
        print('Reconstruction Try, slice id ', idslice)
        data = dxchange.read_tiff_stack(self.args.file_name,ind = range(self.nproj), slc=((idslice,idslice+2,1),(0,self.ni,1))).swapaxes(0,1)#switch to sinogram                 
        data = cp.ascontiguousarray(cp.array(data.astype('float32')))
        rec = cp.zeros([self.nz, self.n, self.n], dtype='float32')
        shift_array = np.arange(-self.args.center_search_width,
                                self.args.center_search_width, self.args.center_search_step).astype('float32')

        with cp.cuda.Stream(non_blocking=False):
            rec_cpu_list = self.recon_try(rec, data, shift_array)
        print('wait until all tiffs are saved to')
        file_name_tmp = os.path.dirname(self.args.file_name)

        fnameout = os.path.dirname(file_name_tmp)+'/try_center/'+os.path.basename(file_name_tmp)+'/r_'
        print(f'{fnameout}')
        write_threads = []
        # avoid simultaneous directory creation
        dxchange.write_tiff(
            rec_cpu_list[0], f'{fnameout}{((self.centeri-shift_array[0])):08.2f}', overwrite=True)
        for k in range(1, len(shift_array)):
            write_thread = threading.Thread(target=dxchange.write_tiff,
                                            args=(rec_cpu_list[k],),
                                            kwargs={'fname': f'{fnameout}{((self.centeri-shift_array[k])):08.2f}',
                                                    'overwrite': True})
            write_threads.append(write_thread)
            write_thread.start()
        for thread in write_threads:
            thread.join()
