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

    def __init__(self, n, nproj, nz, ndark, nflat, data_type):

        # precompute parameters for the lp method
        self.Pgl = initsgl.create_gl(n, nproj)
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
        fZnptr = cp.ascontiguousarray(fZn).data.ptr

        lpids = self.Padj.lpids.data.ptr
        wids = self.Padj.wids.data.ptr
        cids = self.Padj.cids.data.ptr
        nlpids = len(self.Padj.lpids)
        nwids = len(self.Padj.wids)
        ncids = len(self.Padj.cids)

        super().__init__(nproj, nz, n, self.Pgl.Nrho, self.Pgl.Ntheta)
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

    def fix_inf_nan(self, data):
        """Fix inf and nan values in projections"""
        data[cp.isnan(data)] = 0
        data[cp.isinf(data)] = 0
        return data


    def recon(self, el):
        """Full reconstruction pipeline for a data chunk"""
        data = self.darkflat_correction(el['data'], el['dark'], el['flat'])
        data = self.minus_log(data)        
        data = self.fix_inf_nan(data)        
        data = self.fbp_filter(data)        
        # reshape to sinograms
        data = cp.ascontiguousarray(data.swapaxes(0, 1))
        obj = cp.zeros([self.nz, self.n, self.n], dtype=mtype)
        self.backprojection(obj.data.ptr, data.data.ptr)
                
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
        data = fid['exchange/data']# why <=4 doesnt work???
        flat = fid['exchange/data_white']
        dark = fid['exchange/data_dark']
        theta = fid['exchange/theta']
        for ids in chunk(range(data.shape[1]), self.nz):
            # print(f'0: {ids[0]} {ids[-1]}')
            el = {}
            el['data'] = data[:,ids]
            el['flat'] = flat[:,ids]
            el['dark'] = dark[:,ids]
            el['theta'] = theta
            el['ids'] = ids          
            self.data_queue.put(el)  
                        
    def thread1(self):
        stream1 = cp.cuda.Stream(non_blocking=False)
        el = {}
        el['data'] = np.zeros([self.nproj, self.nz,self.n],dtype=self.data_type)
        el['dark'] = np.zeros([self.ndark, self.nz,self.n],dtype=self.data_type)
        el['flat'] = np.ones([self.nflat, self.nz,self.n],dtype=self.data_type)
        
        
        el_pinned = {}
        el_pinned['data'] = self.pinned_array(el['data'])
        el_pinned['dark'] = self.pinned_array(el['dark'])
        el_pinned['flat'] = self.pinned_array(el['flat'])
        while self.running:
            el = self.data_queue.get()    
            el_pinned['data'][:,:len(el['ids'])] = el['data']
            el_pinned['dark'][:,:len(el['ids'])] = el['dark']
            el_pinned['flat'][:,:len(el['ids'])] = el['flat']
            el_gpu = {}
            el_gpu['data'] = cp.zeros([self.nproj, self.nz,self.n],dtype=self.data_type)
            el_gpu['dark'] = cp.zeros([self.ndark, self.nz,self.n],dtype=self.data_type)
            el_gpu['flat'] = cp.ones([self.nflat, self.nz,self.n],dtype=self.data_type)
                            
            with stream1: 
                el_gpu['data'].set(el_pinned['data'])      
                el_gpu['dark'].set(el_pinned['dark'])      
                el_gpu['flat'].set(el_pinned['flat'])                  
                stream1.synchronize()
            # cp.cuda.stream.get_current_stream().synchronize()
                
            el_gpu['ids'] = el['ids'].copy()
            
            # print(f"1: {el_gpu['ids'][0]} {el_gpu['ids'][-1]}")
            self.data_gpu_queue.put(el_gpu)              
            if(el_gpu['ids'][-1]==899):
                print('1 done')
                return             
    
    def thread2(self):
        stream2 = cp.cuda.Stream(non_blocking=False)        
            
        while self.running:
            el_gpu = self.data_gpu_queue.get()            
            elo_gpu = {}
            with stream2:
                elo_gpu['obj'] = self.recon(el_gpu) 
            stream2.synchronize()
            elo_gpu['ids'] = el_gpu['ids'].copy()
            # print(f"2: {elo_gpu['ids'][0]} {elo_gpu['ids'][-1]}")
            self.obj_gpu_queue.put(elo_gpu)   
            if(elo_gpu['ids'][-1]==899):
                return             

    
    def thread3(self):        
            stream3 = cp.cuda.Stream(non_blocking=False)        
            obj_pinned = self.pinned_array(np.zeros([self.nz,self.n, self.n],dtype=mtype))
            while self.running:
                with stream3:                
                    elo_gpu = self.obj_gpu_queue.get()                
                    # print(f"3: {elo_gpu['ids'][0]} {elo_gpu['ids'][-1]}")                
                    elo_gpu['obj'].get(out=obj_pinned)                    
                # cp.cuda.stream.get_current_stream().synchronize()
                    obj = obj_pinned.copy()# temp
                    obj.dtype = 'uint16'#???temp
                stream3.synchronize()
                
                write_thread = threading.Thread(target=dxchange.write_tiff_stack,
                                            args = (obj[:len(elo_gpu['ids'])],),
                                            kwargs = {'fname': '/local/ssd/lprec/r',
                                                        'start': elo_gpu['ids'][0],
                                                        'overwrite': True})                    
                write_thread.start()
                self.write_threads.append(write_thread)                
                if(elo_gpu['ids'][-1]==899):
                    return        
        
    def recon_all(self, data, flat, dark, theta, pchunk=16):
        """Reconstruction by splitting data into chunks"""
        
        thread0 = threading.Thread(target=self.thread0,args = ('/local/ssd/286_2_spfp_019.h5',))   
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
        
        # thread3.join()                     
        
        for thread in self.write_threads:
            thread.join()     
        # while self.running or not self.data_queue.empty():
        #     #tic()                            
        #     el = self.data_queue.get()    
        #     el_gpu[v%2] = self.gpu_copy(el)      
        #     if(v>1):
        #         obj_gpu[v%2] = self.recon(el_gpu[(v-1)%2])            
        #     if(v>2):
        #         obj = obj_gpu[(v-2)%2,:len(el['ids'])].get()
        #         obj.dtype = 'uint16'#???
        #         write_thread = threading.Thread(target=dxchange.write_tiff_stack,
        #                                     args = (obj,),
        #                                     kwargs = {'fname': '/local/ssd/lprec/r',
        #                                                 'start': el['ids'][0],
        #                                                 'overwrite': True})            
        #         if write_thread is not None:
        #             write_thread.start()
        #             write_threads.append(write_thread)
        #     v+=1
        # print(f'waiting threads')
        # for thread in write_threads:
        #     thread.join()                
        # print(f'waiting threads done')            
                

    # def minterplp(self, f, g, x, y, xi, ni, mi):
    #     xc = x.astype('int32')
    #     xf = x-xc
    #     yc = y.astype('int32')
    #     yf = y-yc
    #     ix, iy = cp.unravel_index(xi, (ni, mi))
    #     f[:, ix, iy] = g[:, xc, yc]*(1-xf)*(1-yf)+g[:, xc+1, yc] * \
    #         xf*(1-yf)+g[:, xc, yc+1]*(1-xf)*yf+g[:, xc+1, yc+1]*xf*yf

    # def minterpc(self, f, g, x, y, xi, ni, mi, n, m):
    #     xc = x.astype('int32')
    #     xf = x-xc
    #     yc = y.astype('int32')
    #     yf = y-yc
    #     ix, iy = cp.unravel_index(xi, (ni, mi))
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
