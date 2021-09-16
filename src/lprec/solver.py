from lprec import initsgl
from lprec import initsadj
from lprec import initsadj
from lprec.cfunc import cfunc
from lprec.utils import *
from cupyx.scipy.fft import rfft, irfft, rfft2, irfft2
import cupy as cp
import dxchange
import threading
import queue
import h5py
import os

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
        self.fZn = fZn  # keep in class, otherwise collector will remove it
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

        self.data_queue = queue.Queue()
        self.running = True

    def fbp_filter_center(self, data, center):
        """FBP filtering of projections"""
        ne = 3*self.n//2
        t = cp.fft.rfftfreq(ne).astype(mtype)        
        w = t * (1 - t * 2)**3  # parzen
        w = w*cp.exp(2*cp.pi*1j*t*(center-self.n/2)) # center fix       
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

    def recon(self, obj, data, dark, flat, center):
        """Full reconstruction pipeline for a data chunk"""
        data = self.darkflat_correction(data, dark, flat)
        data = self.minus_log(data)        
        data = self.fbp_filter_center(data, center)        
        data = cp.ascontiguousarray(data.swapaxes(0, 1))
        self.backprojection(obj.data.ptr, data.data.ptr,
                            cp.cuda.get_current_stream().ptr)

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
            item['data'] = data[:self.nproj,  k*self.nz:k*self.nz+lchunk[k]]
            item['flat'] = flat[:,  k*self.nz:k*self.nz+lchunk[k]]
            item['dark'] = dark[:,  k*self.nz:k*self.nz+lchunk[k]]
            self.data_queue.put(item)    

    def recon_all(self, fname, center, pchunk=16):
        """GPU reconstruction of data from an h5file by splitting into chunks"""
        # take links to datasets
        fid = h5py.File(fname, 'r')
        data = fid['exchange/data']
        dark = fid['exchange/data_dark']
        flat = fid['exchange/data_white']
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
            np.zeros([2, self.nproj, self.nz, self.n], dtype=self.data_type))
        item_pinned['dark'] = self.pinned_array(
            np.zeros([2, self.ndark, self.nz, self.n], dtype=self.data_type))
        item_pinned['flat'] = self.pinned_array(
            np.ones([2, self.nflat, self.nz, self.n], dtype=self.data_type))

        # gpu memory for data item
        item_gpu = {}
        item_gpu['data'] = cp.empty_like(item_pinned['data'])
        item_gpu['dark'] = cp.empty_like(item_pinned['dark'])
        item_gpu['flat'] = cp.empty_like(item_pinned['flat'])

        # pinned memory for reconstrution
        rec_pinned = self.pinned_array(
            np.zeros([2, self.nz, self.n, self.n], dtype=mtype))
        # gpu memory for reconstrution
        rec = cp.empty_like(rec_pinned)

        # list of threads for parallel writing to hard disk
        write_threads = []

        # streams for overlapping data transfers with computations
        stream1 = cp.cuda.Stream(non_blocking=False)
        stream2 = cp.cuda.Stream(non_blocking=False)
        stream3 = cp.cuda.Stream(non_blocking=False)

        # Conveyor for data cpu-gpu copy and reconstruction
        #nchunk=3
        tic()
        for k in range(nchunk+2):
            if(k > 0 and k < nchunk+1):
                with stream2:  # reconstruction
                    self.recon(rec[(k-1) % 2], item_gpu['data'][(k-1) % 2],
                               item_gpu['dark'][(k-1) % 2], item_gpu['flat'][(k-1) % 2], center)
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

        print(f'Reconstruction time:{toc():.3f}s')
