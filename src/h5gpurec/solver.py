from h5gpurec.lprec import lprec
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

pinned_memory_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

# 'float32'- c++ code needs to be recompiled with commented add_definitions(-DHALF) in CMakeLists
mtype = 'float32'
    
class H5GPURec():
    def __init__(self, n, nproj, nz, ndark, nflat, data_type, center, double_fov, method, theta):
        # Set ^C interrupt to abort and deallocate memory on GPU
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTSTP, self.signal_handler)
        
        if(double_fov==True):
            n = int(2*n-2*(2*center//2))
            nproj//=2        
        
        theta = cp.ascontiguousarray(cp.array(theta.astype('float32'))/180*np.pi)
        # choose reconstruction method
        if method == 'lprec' :                    
            self.cl_rec = lprec.LpRec(n, nproj, nz)   
        if method == 'fourierrec':
            self.cl_rec = fourierrec.FourierRec(n, nproj, nz, theta)   
    
        self.nz = nz
        self.ndark = ndark
        self.nflat = nflat
        self.data_type = data_type
        self.double_fov = double_fov
        self.center = center
        self.data_queue = queue.Queue()        
        self.running = True
    
    def signal_handler(self, sig, frame):
        """Calls abort_scan when ^C or ^Z is typed"""

        print('Abort')
        os.system('kill -9 $PPID')
            
    def fbp_filter_center(self, data, center):
        """FBP filtering of projections"""

        n = data.shape[2]
        ne = 3*n//2
        t = cp.fft.rfftfreq(ne).astype(mtype)        
        w = t * (1 - t * 2)**3  # parzen
        w = w*cp.exp(2*cp.pi*1j*t*(center-n/2)) # center fix       
        data = cp.pad(data,((0,0),(0,0),(ne//2-n//2,ne//2-n//2)),mode='edge')
        data = irfft(
            w*rfft(data, axis=2), axis=2).astype(mtype)  # note: filter works with complex64, however, it doesnt take much time
        data = data[:,:,ne//2-n//2:ne//2+n//2]
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
                
        nproj, nz, n = data.shape
        nproj_pad = nproj
        if pad:
            nproj_pad = nproj + nproj // 8
        xshift = int((nproj_pad - nproj) // 2)

        xfm = DWTForward(J=1, mode='symmetric', wave=wname).cuda()  # Accepts all wave types available to PyWavelets
        ifm = DWTInverse(mode='symmetric', wave=wname).cuda()    
            
        # Wavelet decomposition.
        cc = []
        sli = torch.zeros([nz,1,nproj_pad,n], device='cuda')
        sli[:,0,(nproj_pad - nproj)//2:(nproj_pad + nproj)//2] = torch.as_tensor(data.swapaxes(0,1), device='cuda')
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
            
        data = cp.asarray(sli[:,0,(nproj_pad - nproj)//2:(nproj_pad + nproj)//2, :n]).swapaxes(0,1)
        return data
    
    def flip_stitch(self, data):
        """Flip and stitch for processing 360degree data with rotation axis on the left border"""

        [nproj, nz, n] = data.shape
        data_new = cp.zeros([nproj,nz,n],dtype=mtype)
        ni = data.shape[2]
        v = cp.linspace(0,1, int(2*self.center),endpoint=False)        
        v = v**5*(126-420*v+540*v**2-315*v**3+70*v**4)     
        data[:nproj,:,:2*center]*=v        
        data_new[:,:,-ni:] = data[:nproj]
        data_new[:,:,:ni] = data[nproj:,:,::-1]      
        return data_new
        
    def recon(self, obj, data, dark, flat):
        """Full reconstruction pipeline for a data chunk"""

        data = self.darkflat_correction(data, dark, flat)
        if(self.double_fov==True):
            data = self.flip_stitch(data)
            center = int(data.shape[2]//2)
        else:
            center = self.center
        data = self.remove_stripe_fw_gpu(data)        
        data = self.minus_log(data)        
        data = self.fbp_filter_center(data, center)        
        data = cp.ascontiguousarray(data.swapaxes(0, 1))
        self.cl_rec.backprojection(obj, data, cp.cuda.get_current_stream())

    def pinned_array(self, array):
        """Allocate pinned memory and associate it with numpy array"""

        mem = cp.cuda.alloc_pinned_memory(array.nbytes)
        src = np.frombuffer(
            mem, array.dtype, array.size).reshape(array.shape)
        src[...] = array
        return src

    def read_data(self, data, dark, flat, nchunk, lchunk):
        """Reading data from hard disk and putting it to a queue"""

        for k in range(nchunk):
            item = {}
            item['data'] = data[:,  k*self.nz:k*self.nz+lchunk[k]]
            item['flat'] = flat[:,  k*self.nz:k*self.nz+lchunk[k]]
            item['dark'] = dark[:,  k*self.nz:k*self.nz+lchunk[k]]
            self.data_queue.put(item)    
            
    def recon_all(self, fname, pchunk=16):
        """GPU reconstruction of data from an h5file by splitting into chunks"""

        # take links to datasets
        fid = h5py.File(fname, 'r')
        data = fid['exchange/data']
        dark = fid['exchange/data_dark']
        flat = fid['exchange/data_white']
        nproji = data.shape[0]
        ni = data.shape[2]

        nchunk = int(np.ceil(data.shape[1]/self.nz))  # number of chunks
        lchunk = np.minimum(
            self.nz, data.shape[1]-np.arange(nchunk)*self.nz)  # chunk sizes

        # start reading data to a queue
        read_thread = threading.Thread(
            target=self.read_data, args=(data, dark, flat, nchunk, lchunk))
        read_thread.start()

        # pinned memory for data item
        item_pinned = {}
        item_pinned['data'] = self.pinned_array(
            np.zeros([2, nproji, self.nz, ni], dtype=self.data_type))
        item_pinned['dark'] = self.pinned_array(
            np.zeros([2, self.ndark, self.nz, ni], dtype=self.data_type))
        item_pinned['flat'] = self.pinned_array(
            np.ones([2, self.nflat, self.nz, ni], dtype=self.data_type))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.zeros([2, nproji, self.nz, ni], dtype=self.data_type)
        item_gpu['dark'] = cp.zeros([2, self.ndark, self.nz, ni], dtype=self.data_type)
        item_gpu['flat'] = cp.ones([2, self.nflat, self.nz, ni], dtype=self.data_type)

        # pinned memory for reconstrution
        if(self.double_fov):
            n = 2*ni
        else:
            n = ni

        rec_pinned = self.pinned_array(
            np.zeros([2, self.nz, n, n], dtype=mtype))
        # gpu memory for reconstrution
        rec = cp.zeros([2, self.nz, n, n], dtype=mtype)

        # list of threads for parallel writing to hard disk
        write_threads = []

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)

        # Conveyor for data cpu-gpu copy and reconstruction
        for k in range(nchunk+2):
            printProgressBar(k,nchunk+1,self.data_queue.qsize(), length = 40)#, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r")
            
            # print(k)
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
                                                      kwargs={'fname': fname[:-3]+'_lprec/r',
                                                              'start': (k-2)*self.nz,
                                                              'overwrite': True})
                write_threads.append(write_thread)
                write_thread.start()
            stream1.synchronize()
            stream2.synchronize()

        # wait until reconstructions are written to hard disk        
        for thread in write_threads:
            thread.join()
