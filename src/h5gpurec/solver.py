from h5gpurec import lprec
from h5gpurec.fourierrec import fourierrec
from h5gpurec.utils import *
from cupyx.scipy.fft import rfft, irfft, rfft2, irfft2
import cupy as cp
import dxchange
import threading
import queue
import h5py
import os
import torch
import signal
import sys
from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
import numexpr as ne

pinned_memory_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

# 'float32'- c++ code needs to be recompiled with commented add_definitions(-DHALF) in CMakeLists
mtype = 'float32'
    
class H5GPURec():
    def __init__(self, args):
        # Set ^C interrupt to abort and deallocate memory on GPU
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)
        
        with h5py.File(args.file_name) as fid:
            [nproj,_,ni] = fid['/exchange/data'].shape
            ndark = fid['/exchange/data_dark'].shape[0]
            nflat = fid['/exchange/data_white'].shape[0]    
            theta = fid['/exchange/theta'][:]        
        # define chunk size for processing            
        nz = args.nsino_per_chunk        
        if(args.reconstruction_type=='try'):
            nz = 2
        # take center
        centeri = args.rotation_axis
        if centeri == -1:
            centeri = ni/2
        # update sizes wrt binning
        ni //= 2**args.binning
        centeri /= 2**args.binning
        
        if(args.file_type=='double_fov'):
            n = 2*ni
            if(centeri<ni//2):
                #if rotation center is on the left side of the ROI
                center = ni-centeri
            else:
                center = centeri                    
        else:
            n = ni            
            center = centeri
        theta = cp.ascontiguousarray(cp.array(theta.astype('float32'))/180*np.pi)
        # choose reconstruction method
        if args.reconstruction_algorithm == 'lprec' :                    
            self.cl_rec = lprec.LpRec(n, nproj, nz)   
        if args.reconstruction_algorithm == 'fourierrec':
            self.cl_rec = fourierrec.FourierRec(n, nproj, nz, theta)   
    
        self.nz = nz
        self.n = n
        self.nproj = nproj
        self.center = center
        self.ni = ni
        self.centeri = centeri
        
        self.ndark = ndark
        self.nflat = nflat
        self.args = args

        self.data_queue = queue.Queue()        
        self.running = True
    
    def signal_handler(self, sig, frame):
        """Calls abort_scan when ^C or ^Z is typed"""

        print('Abort')
        os.system('kill -9 $PPID')
            
    def fbp_filter_center(self, data, sh = 0):
        """FBP filtering of projections"""

        ne = 3*self.n//2
        t = cp.fft.rfftfreq(ne).astype(mtype)        
        w = t * (1 - t * 2)**3  # parzen
        w = w*cp.exp(2*cp.pi*1j*t*(self.center+sh-self.n/2)) # center fix       
        data = cp.pad(data,((0,0),(0,0),(ne//2-self.n//2,ne//2-self.n//2)),mode='edge')
        
        data = irfft(
            w*rfft(data, axis=2), axis=2).astype(mtype)  # note: filter works with complex64, however, it doesnt take much time
        data = data[:,:,ne//2-self.n//2:ne//2+self.n//2]
        
        return data

    def darkflat_correction(self, data, dark, flat):
        """Dark-flat field correction"""

        dark0 = cp.mean(dark, axis=0).astype(mtype)
        flat0 = cp.mean(flat, axis=0).astype(mtype)
        data = (data.astype(mtype)-dark0)/cp.maximum(flat0-dark0, 1e-6)
        return data
    
    def minus_log(self, data):
        """Taking negative logarithm"""
        data = -cp.log(cp.maximum(data, 1e-6))
        return data
    
    def remove_stripe_fw_gpu(self, data):
        """Remove stripes with wavelet filtering"""       

        level = 7#int(np.ceil(np.log2(max(data.shape))))
        wname = 'sym16'
        sigma = 2
        pad = True
                
        nproj_pad = self.nproj
        if pad:
            nproj_pad = self.nproj + self.nproj // 8
        xshift = int((nproj_pad - self.nproj) // 2)

        xfm = DWTForward(J=1, mode='symmetric', wave=wname).cuda()  # Accepts all wave types available to PyWavelets
        ifm = DWTInverse(mode='symmetric', wave=wname).cuda()    
            
        # Wavelet decomposition.
        cc = []
        sli = torch.zeros([self.nz,1,nproj_pad,self.ni], device='cuda')
        sli[:,0,(nproj_pad - self.nproj)//2:(nproj_pad + self.nproj)//2] = torch.as_tensor(data.swapaxes(0,1), device='cuda')
        for k in range(level):
            sli, c = xfm(sli)
            cc.append(c)        
            # FFT
            fcV = torch.fft.fft(cc[k][0][:,0,1],axis=1)
            _, my, mx = fcV.shape
            #Damping of ring artifact information.
            y_hat = torch.fft.ifftshift((torch.arange(-my, my, 2).cuda() + 1) / 2)
            damp = -torch.expm1(-y_hat**2 / (2 * sigma**2))
            fcV *= torch.transpose(torch.tile(damp, (mx, 1)),0,1)
            # Inverse FFT.        
            cc[k][0][:,0,1] = torch.fft.ifft(fcV, my, axis=1).real
            
        # Wavelet reconstruction.
        for k in range(level)[::-1]:
            shape0 = cc[k][0][0,0,1].shape
            sli = sli[:,:,:shape0[0], :shape0[1]]
            sli = ifm((sli, cc[k]))
            
        data = cp.asarray(sli[:,0,(nproj_pad - self.nproj)//2:(nproj_pad + self.nproj)//2, :self.ni]).swapaxes(0,1)
        return data
    
    def pad360(self, data):
        """Pad data with 0 to handle 360 degrees scan"""

        if(self.centeri<self.ni//2):
            #if rotation center is on the left side of the ROI
            data[:] = data[:,:,::-1]            
        w = max(1,int(2*(self.ni-self.center)))    
        v =  cp.linspace(1,0,w,endpoint=False)
        v = v**5*(126-420*v+540*v**2-315*v**3+70*v**4)     
        data[:,:,-w:] *= v            
        
        # double sinogram size with adding 0
        data = cp.pad(data,((0,0),(0,0),(0,data.shape[-1])),'constant')            
        return data
        
    def recon(self, obj, data, dark, flat):
        """Full reconstruction pipeline for a data chunk"""

        data = self.darkflat_correction(data, dark, flat)
        
        if(self.args.remove_stripe_method=='fw'):
            data = self.remove_stripe_fw_gpu(data)        
        data = self.minus_log(data)        
        if(self.args.file_type=='double_fov'):
            data = self.pad360(data)                            
        data = self.fbp_filter_center(data)        
        data = cp.ascontiguousarray(data.swapaxes(0, 1))
        self.cl_rec.backprojection(obj, data, cp.cuda.get_current_stream())

    def pinned_array(self, array):
        """Allocate pinned memory and associate it with numpy array"""

        mem = cp.cuda.alloc_pinned_memory(array.nbytes)
        src = np.frombuffer(
            mem, array.dtype, array.size).reshape(array.shape)
        src[...] = array
        return src

    def downsample(self,data):
        data = data.astype('float32')    
        for j in range(self.args.binning):
            x = data[:,:,::2]
            y = data[:,:,1::2]        
            data = ne.evaluate('x + y')  #should use multithreading          
        for k in range(self.args.binning):            
            x = data[:,::2]
            y = data[:,1::2]        
            data = ne.evaluate('x + y')                        
        return data

    def read_data(self, data, dark, flat, nchunk, lchunk):
        """Reading data from hard disk and putting it to a queue"""

        for k in range(nchunk):
            item = {}
            st =  k*self.nz*2**self.args.binning
            end = (k*self.nz+lchunk[k])*2**self.args.binning
            item['data'] = self.downsample(data[:,  st:end])
            item['flat'] = self.downsample(flat[:,  st:end])
            item['dark'] = self.downsample(dark[:,  st:end])
            self.data_queue.put(item)    
            
    def recon_all(self):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        # take links to datasets
        fid = h5py.File(self.args.file_name, 'r')
        data = fid['exchange/data']
        dark = fid['exchange/data_dark']
        flat = fid['exchange/data_white']
        
        nchunk = int(np.ceil(data.shape[1]/2**self.args.binning/self.nz))  # number of chunks
        lchunk = np.minimum(
            self.nz, np.int32(data.shape[1]/2**self.args.binning-np.arange(nchunk)*self.nz))  # chunk sizes        
        # start reading data to a queue
        read_thread = threading.Thread(
            target=self.read_data, args=(data, dark, flat, nchunk, lchunk))
        read_thread.start()

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = self.pinned_array(
            np.zeros([2, self.nproj, self.nz, self.ni], dtype='float32'))
        item_pinned['dark'] = self.pinned_array(
            np.zeros([2, self.ndark, self.nz, self.ni], dtype='float32'))
        item_pinned['flat'] = self.pinned_array(
            np.ones([2, self.nflat, self.nz, self.ni], dtype='float32'))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros([2, self.nproj, self.nz, self.ni], dtype='float32')
        item_gpu['dark'] = cp.zeros([2, self.ndark, self.nz, self.ni], dtype='float32')
        item_gpu['flat'] = cp.ones([2, self.nflat, self.nz, self.ni], dtype='float32')

        # pinned memory for reconstrution        
        rec_pinned = self.pinned_array(
            np.zeros([2, self.nz, self.n, self.n], dtype=mtype))
        # gpu memory for reconstrution
        rec = cp.zeros([2, self.nz, self.n, self.n], dtype=mtype)

        # list of threads for parallel writing to hard disk
        write_threads = []

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)
        fnameout = os.path.dirname(self.args.file_name)+'_recgpu/'+os.path.basename(self.args.file_name)[:-3]+'_rec/r'
        print('Reconstruction Full')
        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nchunk+2):
            printProgressBar(k,nchunk+1,self.data_queue.qsize(), length = 40)#, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r")            
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
                rec_pinned0 = rec_pinned[(k-2) % 2,:lchunk[k-2],::-1].copy()#.astype('float32')#::-1 is to adapt for tomopy
                #print(np.linalg.norm(rec_pinned0))
                write_thread = threading.Thread(target=dxchange.write_tiff_stack,
                                                      args=(rec_pinned0,),
                                                      kwargs={'fname': fnameout,
                                                              'start': (k-2)*self.nz,
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



    def recon_try(self, obj, data, dark, flat,shift_array):
        """Full reconstruction pipeline for 1 slice with different centers"""

        data = self.darkflat_correction(data, dark, flat)
        
        if(self.args.remove_stripe_method=='fw'):
            data = self.remove_stripe_fw_gpu(data)        
        data = self.minus_log(data)        
        if(self.args.file_type=='double_fov'):
            data = self.pad360(data)                    
        rec_cpu_list=[]
        data0 = data.copy()
        for k in range(len(shift_array)):                    
            printProgressBar(k,len(shift_array)-1,0, length = 40)#, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r")            
            data = self.fbp_filter_center(data0, shift_array[k])                        
            data = cp.ascontiguousarray(data.swapaxes(0, 1))        
            self.cl_rec.backprojection(obj, data, cp.cuda.get_current_stream())            
            rec_cpu_list.append(obj[0].get())
        return rec_cpu_list 
        

    def recon_all_try(self):
        """GPU reconstruction of 1 slice from an h5file"""
        print('Reconstruction Try')
        # take links to datasets
        fid = h5py.File(self.args.file_name, 'r')
        data = fid['exchange/data']
        dark = fid['exchange/data_dark']
        flat = fid['exchange/data_white']
        idslice = int(self.args.nsino*(data.shape[1]-1))
        data = cp.ascontiguousarray(cp.array(data[:,idslice:idslice+2**self.args.binning].astype('float32')))       
        dark = cp.ascontiguousarray(cp.array(dark[:,idslice:idslice+2**self.args.binning].astype('float32')))
        flat = cp.ascontiguousarray(cp.array(flat[:,idslice:idslice+2**self.args.binning].astype('float32')))
        for k in range(self.args.binning):
            data = data[:,:,::2]+data[:,:,1::2]
            dark = dark[:,:,::2]+dark[:,:,1::2]
            flat = flat[:,:,::2]+flat[:,:,1::2]
        for k in range(self.args.binning):
            data = data[:,::2]+data[:,1::2]
            dark = dark[:,::2]+dark[:,1::2]
            flat = flat[:,::2]+flat[:,1::2]
        rec = cp.zeros([self.nz, self.n, self.n], dtype=mtype)
        shift_array = np.arange(-self.args.center_search_width,self.args.center_search_width,self.args.center_search_step).astype('float32')
        
        with cp.cuda.Stream(non_blocking=False):
            rec_cpu_list = self.recon_try(rec,data,dark,flat,shift_array)        
        
        print('wait until all tiffs are saved to')
        fnameout = os.path.dirname(self.args.file_name)+'_recgpu/try_center/'+os.path.basename(self.args.file_name)[:-3]+'/r_'        
        print(f'{fnameout}')
        write_threads=[]
        dxchange.write_tiff(rec_cpu_list[0], f'{fnameout}{(self.centeri-shift_array[0]):08.2f}', overwrite=True)#avoid simultaneous directory creation
        for k in range(1,len(shift_array)):
            write_thread = threading.Thread(target=dxchange.write_tiff,
                                                      args=(rec_cpu_list[k],),
                                                      kwargs={'fname': f'{fnameout}{(self.centeri-shift_array[k]):08.2f}',                                                              
                                                              'overwrite': True})
            write_threads.append(write_thread)
            write_thread.start()
        for thread in write_threads:
            thread.join()

