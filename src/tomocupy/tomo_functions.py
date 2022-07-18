from tomocupy import fourierrec
from tomocupy import lprec
from tomocupy import fbp_filter
from tomocupy import line_summation
from tomocupy import retrieve_phase, remove_stripe


import cupy as cp
import numpy as np
import numexpr as ne
from cupyx.scipy import ndimage

class TomoFunctions():
    def __init__(self, cl_conf):

        self.args = cl_conf.args
        self.ni = cl_conf.ni
        self.n = cl_conf.n
        self.nz = cl_conf.nz
        self.ncz = cl_conf.ncz
        self.nproj = cl_conf.nproj
        self.ncproj = cl_conf.ncproj
        self.centeri = cl_conf.centeri
        self.center = cl_conf.center

        if self.args.reconstruction_algorithm == 'fourierrec':
            self.cl_filter = fbp_filter.FBPFilter(
                self.n, self.nproj, self.ncz, self.args.dtype)
            self.cl_rec = fourierrec.FourierRec(
                self.n, self.nproj, self.ncz, cp.array(cl_conf.theta), self.args.dtype)
        elif self.args.reconstruction_algorithm == 'lprec':            
            self.cl_filter = fbp_filter.FBPFilter(
                self.n, self.nproj, self.ncz, self.args.dtype)
            self.cl_rec = lprec.LpRec(
                self.n, self.nproj, self.ncz, cp.array(cl_conf.theta), self.args.dtype)
        else:
            self.cl_filter = fbp_filter.FBPFilter(
                self.n, self.ncproj, self.nz, self.args.dtype) # note ncproj,nz!
            self.cl_rec = line_summation.LineSummation(
                self.nproj, self.ncproj, self.nz, self.ncz, self.n, self.args.dtype)

    def darkflat_correction(self, data, dark, flat):
        """Dark-flat field correction"""

        dark0 = cp.mean(dark.astype(self.args.dtype,copy=False), axis=0)
        flat0 = cp.mean(flat.astype(self.args.dtype,copy=False), axis=0)
        res = (data.astype(self.args.dtype,copy=False)-dark0)/(flat0-dark0+1e-3)
        res[res<=0] = 1
        return res

    def minus_log(self, data):
        """Taking negative logarithm"""

        data[:] = -cp.log(data)
        data[cp.isnan(data)] = 6.0
        data[cp.isinf(data)] = 0
        return data  # reuse input memory
    
    def remove_outliers(self, data):
        """Remove outliers"""
        
        if(int(self.args.dezinger)>0):
            r = int(self.args.dezinger)            
            fdata = ndimage.median_filter(data,[1,r,r])
            ids = cp.where(cp.abs(fdata-data)>0.5*cp.abs(fdata))
            data[ids] = fdata[ids]    
        return data

    def fbp_filter_center(self, data, sht=0):
        """FBP filtering of projections with applying the rotation center shift wrt to the origin"""

        ne = 3*self.n//2
        if self.args.dtype == 'float16':
            # power of 2 for float16
            ne = 2**int(np.ceil(np.log2(3*self.n//2)))
        t = cp.fft.rfftfreq(ne).astype('float32')
        if self.args.fbp_filter == 'parzen':
            w = t * (1 - t * 2)**3
        elif self.args.fbp_filter == 'shepp':
            w = t * cp.sinc(t)

        
        # beta = 0.007
        # beta = beta*((1-beta)/(1+beta))**cp.arange(len(t))
        # beta = beta.astype('float32')
        # import matplotlib.pyplot as plt
        # plt.plot(w.get())
        # plt.show()
        # exit()
        # beta*=0
        # beta[-1]=1
        # print(beta)
        # alpha = 0.001
        # alpha = 4*beta**2/(1-beta**2)
        # print(alpha)
        # exit()
        # alpha = alpha/cp.sqrt(alpha*(4+alpha))*((2+alpha-cp.sqrt(alpha*(4+alpha)))/2)**cp.arange(len(t))
        # alpha = alpha.astype('float32')
        # print(alpha)
        # exit(1)
        w = w*cp.exp(-2*cp.pi*1j*t*(-self.center +
                     sht[:, cp.newaxis]+self.n/2))  # center fix
        tmp = cp.pad(
            data, ((0, 0), (0, 0), (ne//2-self.n//2, ne//2-self.n//2)), mode='edge')
        # tmp = cp.fft.irfft(
            # beta*cp.fft.rfft(tmp, axis=2), axis=2).astype(self.args.dtype)  # note: filter works with complex64, however, it doesnt take much time
        self.cl_filter.filter(tmp, w, cp.cuda.get_current_stream())
        data[:] = tmp[:, :, ne//2-self.n//2:ne//2+self.n//2]

        return data  # reuse input memory

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

    def proc_sino(self, data, dark, flat, res=None):
        """Processing a sinogram data chunk"""

        if not isinstance(res, cp.ndarray):
            res = cp.zeros(data.shape, self.args.dtype)
        # dark flat field correrction
        res[:] = self.darkflat_correction(data, dark, flat)
        res[:] = self.remove_outliers(res)
        # remove stripes
        if(self.args.remove_stripe_method == 'fw'):
            res[:] = remove_stripe.remove_stripe_fw(
                res, self.args.fw_sigma, self.args.fw_filter, self.args.fw_level)
            
        return res

    def proc_proj(self, data, res=None):
        """Processing a projection data chunk"""

        if not isinstance(res, cp.ndarray):
            res = cp.zeros(
                [data.shape[0], data.shape[1], self.n], self.args.dtype)
        # retrieve phase
        if self.args.retrieve_phase_method == 'paganin':
            data[:] = retrieve_phase.paganin_filter(
                data,  self.args.pixel_size*1e-4, self.args.propagation_distance/10, self.args.energy, self.args.retrieve_phase_alpha)
        # minus log
        data[:] = self.minus_log(data)
        # padding for 360 deg recon
        if(self.args.file_type == 'double_fov'):
            res[:] = self.pad360(data)
        else:
            res[:] = data[:]
        return res
