from lprec import initsgl
from lprec import initsadj
from lprec import initsadj
from lprec.cfunc import cfunc
from lprec.utils import *
import cupy as cp
from cupyx.scipy.fft import rfft, irfft, rfft2, irfft2
import cupyx
import dxchange
import threading
import queue
import h5py
import time

pinned_memory_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

# 'float32'- c++ code needs to be recompiled with changed directives in cfunc.cuh
mtype = 'float16'
class LpRec(cfunc):

    def __init__(self, n, nproj, nz, ntheta, nrho, ndark, nflat, data_type):

        # precompute parameters for the lp method
        self.Pgl = initsgl.create_gl(n, nproj, ntheta, nrho)
        self.Padj = initsadj.create_adj(self.Pgl)
        lp2p1 = self.Padj.lp2p1.data.ptr
        lp2p2 = self.Padj.lp2p2.data.ptr
        lp2p1w = self.Padj.lp2p1w.data.ptr
        lp2p2w = self.Padj.lp2p2w.data.ptr
        C2lp1 = self.Padj.C2lp1.data.ptr
        C2lp2 = self.Padj.C2lp2.data.ptr

        # conevrt fZ from complex to float.. coudl be improved..
        fZn = cp.zeros(
            [self.Padj.fZ.shape[0], self.Padj.fZ.shape[1]*2], dtype=mtype)
        fZn[:, ::2] = self.Padj.fZ.real
        fZn[:, 1::2] = self.Padj.fZ.imag
        self.fZn = fZn # keep in class, otherwise collector will remove it
        fZnptr = cp.ascontiguousarray(self.fZn).data.ptr

        lpids = self.Padj.lpids.data.ptr
        wids = self.Padj.wids.data.ptr
        cids = self.Padj.cids.data.ptr
        nlpids = len(self.Padj.lpids)
        nwids = len(self.Padj.wids)
        ncids = len(self.Padj.cids)

        super().__init__(nproj, nz, n, ntheta, nrho)
        super().setgrids(fZnptr, lp2p1, lp2p2, lp2p1w, lp2p2w,
                         C2lp1, C2lp2, lpids, wids, cids,
                         nlpids, nwids, ncids)
        self.ndark = ndark
        self.nflat = nflat
        self.data_type = data_type
                    
        self.write_threads = []
        self.data_queue = queue.Queue()   
        self.data_gpu_queue = queue.Queue(maxsize=1)   
        self.obj_gpu_queue = queue.Queue(maxsize=1)   
        self.running = True
            

    def fbp_filter(self, data):
        """FBP filtering of projections"""
        w = cp.tile(self.Padj.wfilter, [data.shape[1], 1])
        data = irfft(
            w*rfft(data, overwrite_x=True, axis=2), overwrite_x=True, axis=2).astype(mtype)  # note: filter works with complex64, however, it doesnt take much time
        return data

    def darkflat_correction(self, data, dark, flat):
        """Dark-flat field correction"""
        dark0 = cp.mean(dark,axis=0).astype(mtype)
        flat0 = cp.mean(flat,axis=0).astype(mtype)                
        data = (data.astype(mtype)-dark0)/cp.maximum(flat0-dark0, 1e-6)
        return data

    def minus_log(self, data):
        """Taking negative logarithm"""
        data = -cp.log(cp.maximum(data, 1e-6))
        return data

    def recon(self, item):
        """Full reconstruction pipeline for a data chunk"""
        data = self.darkflat_correction(item['data'], item['dark'], item['flat'])
        data = self.minus_log(data)        
        data = self.fbp_filter(data)     
        obj = cp.zeros([self.nz, self.n, self.n], dtype=mtype)        
        data = cp.ascontiguousarray(data.swapaxes(0, 1))
        self.backprojection(obj.data.ptr, data.data.ptr, cp.cuda.get_current_stream().ptr)
                
        return obj
            
    def pinned_array(self,array):
        # first constructing pinned memory
        mem = cp.cuda.alloc_pinned_memory(array.nbytes)
        src = np.frombuffer(
                    mem, array.dtype, array.size).reshape(array.shape)
        src[...] = array
        return src
    
    def thread0(self, file_name):
        fid = h5py.File(file_name, 'r')
        data = fid['exchange/data']
        flat = fid['exchange/data_white']
        dark = fid['exchange/data_dark']
        #theta = fid['exchange/theta']
        for ids in chunk(range(data.shape[1]), self.nz):
            print(f'0: {ids[0]} {ids[-1]}')
            item = {}
            item['data'] = np.pad(data[:-1,ids],((0,0),(0,self.nz-len(ids)),(0,0)))
            item['flat'] = np.pad(flat[:,ids],((0,0),(0,self.nz-len(ids)),(0,0)))
            item['dark'] = np.pad(dark[:,ids],((0,0),(0,self.nz-len(ids)),(0,0)))
            item['ids'] = ids.copy()          
            self.data_queue.put(item)  
            if(item['ids'][-1]==1791):
                return   
                        
    def thread1(self):
        stream1 = cp.cuda.Stream()        
        item_pinned = {}
        item_pinned['data'] = self.pinned_array(np.zeros([self.nproj, self.nz,self.n],dtype=self.data_type))
        item_pinned['dark'] = self.pinned_array(np.zeros([self.ndark, self.nz,self.n],dtype=self.data_type))
        item_pinned['flat'] = self.pinned_array(np.ones([self.nflat, self.nz,self.n],dtype=self.data_type))
        while self.running:
            item_gpu = {}
            item_gpu['data'] = cp.empty_like(item_pinned['data'])
            item_gpu['dark'] = cp.empty_like(item_pinned['dark'])
            item_gpu['flat'] = cp.empty_like(item_pinned['flat'])
            with stream1:             
                item = self.data_queue.get()                    
                item_pinned['data'][:] = item['data']
                item_pinned['dark'][:] = item['dark']
                item_pinned['flat'][:] = item['flat']                                
                item_gpu['data'].set(item_pinned['data'])      
                item_gpu['dark'].set(item_pinned['dark'])      
                item_gpu['flat'].set(item_pinned['flat'])                  
                item_gpu['ids'] = item['ids'].copy()            
            stream1.synchronize()                
            
            print(f"1: {item_gpu['ids'][0]} {item_gpu['ids'][-1]}")
            self.data_gpu_queue.put(item_gpu)              
            if(item_gpu['ids'][-1]==1791):
                return             
    
    def thread2(self):
        stream2 = cp.cuda.Stream()        
        while self.running:
            item_obj_gpu = {}                                                
            with stream2:                
                item_gpu = self.data_gpu_queue.get()            
                item_obj_gpu['obj'] = self.recon(item_gpu) 
                item_obj_gpu['ids'] = item_gpu['ids'].copy()            
            stream2.synchronize()                            
            print(f"2: {item_obj_gpu['ids'][0]} {item_obj_gpu['ids'][-1]}")                        
            self.obj_gpu_queue.put(item_obj_gpu)   
            if(item_gpu['ids'][-1]==1791):
                return             
    
    def thread3(self):        
        stream3 = cp.cuda.Stream()        
        obj_pinned = self.pinned_array(np.zeros([self.nz,self.n, self.n],dtype=mtype))
        while self.running:
            with stream3:                
                item_gpu = self.obj_gpu_queue.get()                
                item_gpu['obj'].get(out=obj_pinned)                    
                obj = obj_pinned.copy()# temp
                # obj.dtype = 'uint16'#???temp
            stream3.synchronize()
            print(f"3: {item_gpu['ids'][0]} {item_gpu['ids'][-1]}")                                                    
            write_thread = threading.Thread(target=dxchange.write_tiff_stack,
                                        args = (obj[:len(item_gpu['ids'])],),
                                        kwargs = {'fname': '/local/ssd/lprec/r',
                                                    'start': item_gpu['ids'][0],
                                                    'overwrite': True})                    
            write_thread.start()            
            self.write_threads.append(write_thread)                
            if(item_gpu['ids'][-1]==1791):
                # dxchange.write_tiff(obj[0],'/local/ssd/data/tmp/t'+str(self.ntheta)+str(self.nrho),overwrite=True)
                return        
        
    def recon_all(self, fname, pchunk=16):
        """Reconstruction by splitting data into chunks"""
        
        thread0 = threading.Thread(target=self.thread0,args = (fname,))   
        thread1 = threading.Thread(target=self.thread1)   
        thread2 = threading.Thread(target=self.thread2)   
        thread3 = threading.Thread(target=self.thread3)   
        thread0.start()                
        thread1.start()
        thread2.start()
        thread3.start()
        thread0.join()        
        thread1.join()        
        thread2.join()        
        thread3.join()        
        for thread in self.write_threads:
            thread.join()     
    # def minterplp(self, f, g, x, y, xi, ni, mi):
    #     xc = x.astype('int32')
    #     xf = x-xc
    #     yc = y.astype('int32')
    #     yf = y-yc
    #     ix, iy = cp.unravitem_index(xi, (ni, mi))
    #     f[:, ix, iy] = g[:, xc, yc]*(1-xf)*(1-yf)+g[:, xc+1, yc] * \
    #         xf*(1-yf)+g[:, xc, yc+1]*(1-xf)*yf+g[:, xc+1, yc+1]*xf*yf

    # def minterpc(self, f, g, x, y, xi, ni, mi, n, m):
    #     xc = x.astype('int32')
    #     xf = x-xc
    #     yc = y.astype('int32')
    #     yf = y-yc
    #     ix, iy = cp.unravitem_index(xi, (ni, mi))
    #     f[:, ix, iy] += g[:, xc, yc]*(1-xf)*(1-yf)+g[:, (xc+1) % n, yc]*xf*(
    #         1-yf)+g[:, xc, (yc+1) % m]*(1-xf)*yf+g[:, (xc+1) % n, (yc+1) % m]*xf*yf
    #     return f

    # def backprojection2(self, R):
    #     nz = R.shape[0]
    #     f = cp.zeros([nz, self.n, self.n], dtype=mtype)
    #     Nchunk = nz
    #     for ids in chunk(range(nz),Nchunk):
    #         for k in range(3):
    #             Rlp0 = cp.zeros([Nchunk, self.nrho, self.ntheta], dtype=mtype)
    #             self.minterplp(Rlp0, R[ids], self.Padj.lp2p1[k],
    #                             self.Padj.lp2p2[k], self.Padj.lpids, self.nrho, self.ntheta)
    #             self.minterplp(Rlp0, R[ids], self.Padj.lp2p1w[k],
    #                             self.Padj.lp2p2w[k], self.Padj.wids, self.nrho, self.ntheta)
    #             # flp=Rlp0*self.Padj.fZ;
    #             flp = irfft2(rfft2(Rlp0,overwrite_x=True)*self.Padj.fZ,overwrite_x=True)
    #             f[ids] = self.minterpc(f[ids], flp, self.Padj.C2lp2[k],
    #                                     self.Padj.C2lp1[k], self.Padj.cids, self.n, self.n, self.nrho, self.ntheta)
    #     return f
