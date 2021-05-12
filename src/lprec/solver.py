from lprec import initsgl
from lprec import initsadj
from lprec import initsadj
from .cfunc import cfunc
from lprec.timing import *
import cupy as cp
from scipy import interpolate
from itertools import islice
from cupyx.scipy.fft import rfft, irfft, rfft2, irfft2
import h5py
import dxchange
import matplotlib.pyplot as plt

class LpRec(cfunc):
    def __init__(self, n, nproj, nz):
        
        # precompute parameters for the lp method
        self.Pgl = initsgl.create_gl(n, nproj)
        self.Padj = initsadj.create_adj(self.Pgl)
        lp2p1 = self.Padj.lp2p1.data.ptr                
        lp2p2 = self.Padj.lp2p2.data.ptr                
        lp2p1w = self.Padj.lp2p1w.data.ptr                
        lp2p2w = self.Padj.lp2p2w.data.ptr                
        C2lp1 = self.Padj.C2lp1.data.ptr                
        C2lp2 = self.Padj.C2lp2.data.ptr                
        fZ = self.Padj.fZ.data.ptr           
        lpids = self.Padj.lpids.data.ptr           
        wids = self.Padj.wids.data.ptr           
        cids = self.Padj.cids.data.ptr           
        nlpids = len(self.Padj.lpids)
        nwids = len(self.Padj.wids)
        ncids = len(self.Padj.cids)

        super().__init__(nproj, nz, n, self.Pgl.Nrho, self.Pgl.Ntheta)    
        super().setgrids(fZ, lp2p1, lp2p2, lp2p1w, lp2p2w, 
            C2lp1, C2lp2, lpids, wids, cids, 
            nlpids, nwids, ncids)
        
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
    #     f[:, ix, iy] = g[:, xc, yc]*(1-xf)*(1-yf)+g[:, (xc+1) % n, yc]*xf*(
    #         1-yf)+g[:, xc, (yc+1) % m]*(1-xf)*yf+g[:, (xc+1) % n, (yc+1) % m]*xf*yf
    #     return f

    # def backprojection2(self, R):                
    #     nz = R.shape[0]        
    #     f = cp.zeros([nz, self.n, self.n], dtype='float32')        
    #     Nchunk = nz
    #     for ids in chunk(range(nz),Nchunk):
    #         for k in range(3):
    #             Rlp0 = cp.zeros([Nchunk, self.nrho, self.ntheta], dtype='float32')
    #             self.minterplp(Rlp0, R[ids], self.Padj.lp2p2[k],
    #                             self.Padj.lp2p1[k], self.Padj.lpids, self.nrho, self.ntheta)
    #             self.minterplp(Rlp0, R[ids], self.Padj.lp2p2w[k],
    #                             self.Padj.lp2p1w[k], self.Padj.wids, self.nrho, self.ntheta)
    #             flp = irfft2(rfft2(Rlp0,overwrite_x=True)*self.Padj.fZ,overwrite_x=True)
    #             f[ids] += self.minterpc(f[ids], flp, self.Padj.C2lp2[k],
    #                                     self.Padj.C2lp1[k], self.Padj.cids, self.n, self.n, self.nrho, self.ntheta)                
    #     return f


    def fbp_filter(self,data):
        """FBP filtering of projections"""
        t = cp.fft.rfftfreq(data.shape[2])
        wfilter = t * (1 - t * 2)**3  # parzen
        wfilter = cp.tile(wfilter, [data.shape[1], 1])
        # loop over slices to minimize fft memory overhead
        for k in range(data.shape[0]):
            data[k] = irfft(
                wfilter*rfft(data[k], overwrite_x=True, axis=1), overwrite_x=True, axis=1)
        return data


    def darkflat_correction(self, data, dark, flat):
        """Dark-flat field correction"""
        for k in range(data.shape[0]):
            data[k] = (data[k]-dark)/cp.maximum(flat-dark, 1e-6)
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


    def binning(self, data, bin_level):
        for k in range(bin_level):
            data = 0.5*(data[..., ::2, :]+data[..., 1::2, :])
            data = 0.5*(data[..., :, ::2]+data[..., :, 1::2])
        return data


    def gpu_copy(self, data,  dark, flat,  theta, start, end, bin_level):
        data_gpu = cp.array(data[:,start:end]).astype('float32')
        theta_gpu = cp.array(theta).astype('float32')
        data_gpu = self.binning(data_gpu, bin_level)        
        dark_gpu = cp.mean(cp.array(dark[:,start:end]), axis=0).astype('float32')
        dark_gpu = self.binning(dark_gpu, bin_level)
        flat_gpu = cp.mean(cp.array(flat[:,start:end]), axis=0).astype('float32')        
        flat_gpu = self.binning(flat_gpu, bin_level)
        return data_gpu,dark_gpu,flat_gpu,theta_gpu


    def recon(self, data, dark, flat, theta):
        data = self.darkflat_correction(data, dark, flat)
        data = self.minus_log(data)
        data = self.fix_inf_nan(data)
        data = self.fbp_filter(data)
        data = cp.ascontiguousarray(data.swapaxes(0,1))
        #print(data.shape)
        # plt.imshow(data[:,0].get())
        # plt.show()
        obj = cp.zeros([self.nz,self.n,self.n],dtype='float32')
        #print(data.shape)
        # data = cp.array(cp.load('data/R.npy')).swapaxes(1,2)
        # print(data.shape)
        # exit()
        # data = cp.tile(data,[self.nz,1,1])        
        self.backprojection(obj.data.ptr,data.data.ptr)
        #obj1 = self.backprojection2(data.swapaxes(1,2))
        # plt.subplot(131)
        # plt.imshow(obj[0].get())
        # plt.colorbar()
        # plt.subplot(132)
        # plt.imshow(obj1[0].get())
        # plt.subplot(133)
        # plt.imshow((obj1[0]-obj[0]).get())
        # plt.colorbar()
        # plt.show()
        return obj
    
    def recon_all(self, file_name, bin_level=0, pchunk=16):
        fid = h5py.File(file_name, 'r')
        data = fid['exchange/data']
        flat = fid['exchange/data_white']
        dark = fid['exchange/data_dark']
        theta = fid['exchange/theta']
        # compute mean of dark and flat fields on GPU
        
        #nnz = 32
        # nnproj = 3*512//2
        # nn=512
        # data = data[:nnproj,:nnz,:nn]
        # dark = dark[:,:nnz,:nn]
        # flat = flat[:,:nnz,:nn]
        # theta = theta[:nnproj]

        print(self.nz)
        data = data[:-1,:512,:]
        dark = 0*dark[:,:512,:]
        flat = dark[:,:512,:]
        theta = theta[:]
        pchunk = self.nz
        
        for k in range(data.shape[1]//pchunk):
            print(k)
            # thread for cpu-gpu copy
            tic()
            data_gpu, dark_gpu, flat_gpu, theta_gpu = self.gpu_copy(data, dark, flat, theta, k*pchunk, min((k+1)*pchunk, data.shape[1]), bin_level)
            cp.cuda.stream.get_current_stream().synchronize()
            print(f'copy {toc()}')
            # thread for processing
            tic()
            obj_gpu = self.recon(data_gpu, dark_gpu, flat_gpu, theta_gpu)
            cp.cuda.stream.get_current_stream().synchronize()
            print(f'recon {toc()}')
            tic()
            obj=obj_gpu.get()
            cp.cuda.stream.get_current_stream().synchronize()
            print(f'copy back {toc()}')
            dxchange.write_tiff_stack(obj, '/local/data/lprec/r',start=k*pchunk,overwrite=True)

