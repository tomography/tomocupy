from lprec import initsgl
from lprec import initsadj
from lprec import initsadj
from lprec.cfunc import cfunc
from lprec.utils import *
import cupy as cp
from cupyx.scipy.fft import rfft, irfft, rfft2, irfft2

# 'float32'- c++ code needs to be recompiled with changed directives in cfunc.cuh
mtype = 'float16'


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

    def fbp_filter(self, data):
        """FBP filtering of projections"""
        w = cp.tile(self.Padj.wfilter, [data.shape[1], 1])
        data = irfft(
            w*rfft(data, overwrite_x=True, axis=2), overwrite_x=True, axis=2).astype(mtype)  # note: filter works with complex64, however, it doesnt take much time
        return data

    def darkflat_correction(self, data, dark, flat):
        """Dark-flat field correction"""
        data = (data-dark)/cp.maximum(flat-dark, 1e-6)
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
        """Downsample data"""
        for k in range(bin_level):
            data = 0.5*(data[..., ::2, :]+data[..., 1::2, :])
            data = 0.5*(data[..., :, ::2]+data[..., :, 1::2])
        return data

    def gpu_copy(self, data,  dark, flat,  theta, ids, bin_level):
        """Copy data to GPU as mtype arrays"""
        data_gpu = cp.zeros([self.nproj,self.nz,self.n],dtype=mtype)
        dark_gpu = cp.zeros([self.nz,self.n],dtype=mtype)
        flat_gpu = cp.ones([self.nz,self.n],dtype=mtype)
        
        data_gpu[:,:len(ids)] = cp.array(data[:, ids])
        # data_gpu = self.binning(data_gpu, bin_level)
        dark_gpu[:len(ids)] = cp.mean(cp.array(dark[:, ids]), axis=0)
        # dark_gpu = self.binning(dark_gpu, bin_level)
        flat_gpu[:len(ids)] = cp.mean(cp.array(flat[:, ids]), axis=0)
        # flat_gpu = self.binning(flat_gpu, bin_level)                
        theta_gpu = cp.array(theta).astype(mtype)
        
        return data_gpu, dark_gpu, flat_gpu, theta_gpu

    def recon(self, data, dark, flat, theta):
        """Full reconstruction pipeline for a data chunk"""
        data = self.darkflat_correction(data, dark, flat)
        data = self.minus_log(data)        
        data = self.fix_inf_nan(data)        
        data = self.fbp_filter(data)        
        # reshape to sinograms
        data = cp.ascontiguousarray(data.swapaxes(0, 1))
        obj = cp.zeros([self.nz, self.n, self.n], dtype=mtype)
        self.backprojection(obj.data.ptr, data.data.ptr)
        
        # import matplotlib.pyplot as plt
        # obj2=self.backprojection2(data)
        # plt.subplot(131)
        # plt.imshow(obj[0].astype('float32').get(),cmap='gray')
        # plt.colorbar()
        # plt.subplot(132)
        # plt.imshow(obj2[0].astype('float32').get(),cmap='gray')
        # plt.colorbar()
        # plt.subplot(133)
        # dif = np.abs(obj2[0].astype('float32').get()-obj[0].astype('float32').get())
        # plt.imshow(dif,cmap='gray')
        # print(np.linalg.norm(dif))
        # plt.clim([0,dif.max()/10,])
        # plt.colorbar()
        #
        # plt.show()

        return obj

    def recon_all(self, data, flat, dark, theta, bin_level=0, pchunk=16):
        """Reconstruction by splitting data into chunks"""
        obj = np.zeros([data.shape[1],data.shape[2],data.shape[2]],dtype=mtype)        
        for ids in chunk(range(data.shape[1]), self.nz):
            print(f'Reconstruction of slices {ids[0]}..{ids[-1]}')
            # tic()
            # thread for cpu-gpu copy
            tic()
            data_gpu, dark_gpu, flat_gpu, theta_gpu = self.gpu_copy(
                data, dark, flat, theta, ids,  bin_level)
            cp.cuda.stream.get_current_stream().synchronize()
            print(f'copy to GPU {toc():.2f}')
            # thread for processing
            tic()
            obj_gpu = self.recon(data_gpu, dark_gpu, flat_gpu, theta_gpu)
            cp.cuda.stream.get_current_stream().synchronize()
            print(f'recon {toc():.2f}')
            tic()
            obj[ids] = obj_gpu[:len(ids)].get()
            cp.cuda.stream.get_current_stream().synchronize()
            print(f'copy back to CPU {toc():.2f}')
            # print(f'Time {toc():.3f}s')
        
        return obj

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
