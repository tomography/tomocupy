from tomocupy import lprec, fourierrec
from tomocupy import retrieve_phase, remove_stripe
from tomocupy import utils
from cupyx.scipy.fft import rfft, irfft
import cupy as cp
import numpy as np
import dxchange
import threading
import queue
import h5py
import os
import signal
import numexpr as ne

pinned_memory_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

class GPUProc():
    def __init__(self, args):
        # Set ^C interrupt to abort and deallocate memory on GPU
        signal.signal(signal.SIGINT, utils.signal_handler)
        signal.signal(signal.SIGTSTP, utils.signal_handler)

        with h5py.File(args.file_name) as fid:
            # determine sizes
            n = fid['/exchange/data'].shape[2]
            if (args.end_row==-1):
                args.end_row = fid['/exchange/data'].shape[1]
            if (args.end_proj==-1):
                args.end_proj = fid['/exchange/data'].shape[0]
            nz = args.end_row  - args.start_row                    
            nproj = args.end_proj - args.start_proj
            # read angles,dark and flat fields
            theta = fid['/exchange/theta'][:].astype('float32')/180*np.pi
            dark = fid['/exchange/data_dark'][:,args.start_row:args.end_row].astype('float32')
            flat = fid['/exchange/data_white'][:,args.start_row:args.end_row].astype('float32')
                
        ids_proj = np.arange(len(theta))[args.start_proj:args.end_proj]
        theta = theta[ids_proj]
        # blocked views fix        
        if args.blocked_views:
            st = args.blocked_views_start
            end = args.blocked_views_end
            ids = np.where(((theta) % np.pi < st) +
                           ((theta-st) % np.pi > end-st))[0]
            theta = theta[ids]
            ids_proj = ids_proj[ids]            
        
        nproj = len(theta)        
        ncproj = args.nproj_per_chunk
                
        # update sizes wrt binning
        n //= 2**args.binning        
        nz //= 2**args.binning        
        
        self.n = n
        self.nz = nz
        self.nproj = nproj
        self.ncproj = ncproj
        
        self.ids_proj = ids_proj
        self.theta = theta
        self.args = args

        self.dark = cp.array(np.mean(self.downsample(dark),axis=0))
        self.flat = cp.array(np.mean(self.downsample(flat),axis=0))        
        # make sure dark<=flat
        self.dark[self.dark>self.flat] = self.flat
        # queue for streaming projections
        self.data_queue = queue.Queue()


    def downsample(self, data):
        """Downsample data"""

        data = data[:].astype('float32')
        for j in range(self.args.binning):
            x = data[:, :, ::2]
            y = data[:, :, 1::2]
            data = ne.evaluate('x + y')  # should use multithreading
        for k in range(self.args.binning):
            x = data[:, ::2]
            y = data[:, 1::2]
            data = ne.evaluate('x + y')
        return data

    def darkflat_correction(self, data):
        """Dark-flat field correction"""

        data = (data-self.dark)/(self.flat-self.dark+1e-3)
        return data

    def minus_log(self, data):
        """Taking negative logarithm"""

        data = -cp.log(data)
        data[cp.isnan(data)] = 6.0
        data[cp.isinf(data)] = 0
        return data
    
    def read_data(self, data, nchunk, lchunk):
        """Reading data from hard disk and putting it to a queue"""

        for k in range(nchunk):
            item = {}
            st = k*self.ncproj
            end = k*self.ncproj+lchunk[k]        
            st_row = self.args.start_row    
            end_row = self.args.end_row
            #import pdb; pdb.set_trace()
            item['data'] = self.downsample(data[self.ids_proj[st:end],st_row:end_row])
            
            self.data_queue.put(item)

    def proc(self, rec, data):
        """Full processing pipeline for a data chunk"""

        # dark-flat field correction
        data = self.darkflat_correction(data)        
        # remove stripes// doesnt work well if projections are chunked -> do it in the recon sinogram domain
        #if(self.args.remove_stripe_method == 'fw'):
        #    data = remove_stripe.remove_stripe_fw(data)            
        # retrieve phase
        if(self.args.retrieve_phase_method == 'paganin'):
            data = retrieve_phase.paganin_filter(
                data,  self.args.pixel_size*1e-4, self.args.propagation_distance/10, self.args.energy, self.args.retrieve_phase_alpha)
        # minus log
        rec[:] = self.minus_log(data)
        return rec    
   
    def proc_all(self):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        # take links to datasets
        fid = h5py.File(self.args.file_name, 'r')
        
        # init references to data, no reading at this point
        data = fid['exchange/data']
        
        nchunk = int(np.ceil(self.nproj/self.ncproj))
        lchunk = np.minimum(
            self.ncproj, np.int32(self.nproj-np.arange(nchunk)*self.ncproj))  # chunk sizes
        # start reading data to a queue
        read_thread = threading.Thread(
            target=self.read_data, args=(data, nchunk, lchunk))
        read_thread.start()

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = utils.pinned_array(
            np.zeros([2, self.ncproj, self.nz, self.n], dtype='float32'))
        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros(
            [2, self.ncproj, self.nz, self.n], dtype='float32')
        
        # pinned memory for processed data
        rec_pinned = utils.pinned_array(
            np.zeros([2, self.ncproj, self.nz, self.n], dtype='float32'))
        # gpu memory for processed data
        rec = cp.zeros([2, self.ncproj, self.nz, self.n], dtype='float32')

        # list of threads for parallel writing to hard disk
        write_threads = []

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)
        if(self.args.out_path_name is None):
            fnameout = os.path.dirname(
                self.args.file_name)+'_recgpu/'+os.path.basename(self.args.file_name)[:-3]+'_proc'
        else:
            fnameout = str(self.args.out_path_name)
        print('Processing projections')
        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nchunk+2):
            utils.printProgressBar(k, nchunk+1, self.data_queue.qsize(), length=40)
            if(k > 0 and k < nchunk+1):
                with stream2:  # reconstruction
                    self.proc(rec[(k-1) % 2], item_gpu['data'][(k-1) % 2])
                    
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
                rec_pinned0 = rec_pinned[(k-2) % 2, :lchunk[k-2]].copy()                                
                write_thread = threading.Thread(target=dxchange.write_tiff_stack,
                                                args=(rec_pinned0,),
                                                kwargs={'fname': fnameout+'/p',
                                                        'start': (k-2)*self.ncproj,
                                                        'overwrite': True})                
                                             
                write_threads.append(write_thread)
                write_thread.start()                
            stream1.synchronize()
            stream2.synchronize()
        print(f'Output folder: {fnameout}')
        print(f'Saving angles as {fnameout}/angles.npy')
        np.save(fnameout+'/angles.npy',self.theta)
        
        # wait until reconstructions are written to hard disk
        for thread in write_threads:
            thread.join()
        

