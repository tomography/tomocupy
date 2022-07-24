"""
CUDA Raw kernels for computing back-projection
"""

import cupy as cp

source = """
extern "C" {        
    void __global__ backprojection(float *f, float *data, float *theta, float phi, int sz, float c, int ncz, int n, int nz, int nproj)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= n || tz >= ncz)
            return;
        float u = 0;
        float v = 0;
        int ur = 0;
        int vr = 0;        
        
        float f0 = 0;
        float theta0 = 0;
        float cphi = __cosf(phi);
        float sphi = __sinf(phi);
        float R[6] = {};
        
        for (int t = 0; t<nproj; t++)
        {
            theta0 = theta[t];            
            float ctheta = __cosf(theta0);
            float stheta = __sinf(theta0);
            R[0] =  ctheta;       R[1] =  stheta;        R[2] = 0;
            R[3] =  stheta*cphi;  R[4] = -ctheta*cphi;   R[5] = sphi;
            u = R[0]*(tx-n/2)+R[1]*(ty-n/2)+n/2;
            v = R[3]*(tx-n/2)+R[4]*(ty-n/2)+R[5]*(tz+sz-nz/2) + nz/2;
            
            ur = (int)(u-1e-5f);
            vr = (int)(v-1e-5f);            
            
            // linear interp            
            if ((ur >= 0) & (ur < n - 1) & (vr >= 0) & (vr < nz-1))
            {
                u = u-ur;
                v = v-vr;                
                f0 +=   data[ur+0+t*n+(vr+0)*n*nproj]*(1-u)*(1-v)+
                        data[ur+1+t*n+(vr+0)*n*nproj]*(0+u)*(1-v)+
                        data[ur+0+t*n+(vr+1)*n*nproj]*(1-u)*(0+v)+
                        data[ur+1+t*n+(vr+1)*n*nproj]*(0+u)*(0+v);
            }
        }
        f[tx + (n-ty-1) * n + tz * n * n] += f0*c;        
    }    

    void __global__ backprojection_try(float *f, float *data, float *theta, float phi, int sz, float* sh, float c, int ncz, int n, int nz, int nproj)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= n || tz >= ncz)
            return;
        float u = 0;
        float v = 0;
        int ur = 0;
        int vr = 0;        
        
        float f0 = 0;
        float theta0 = 0;
        float cphi = __cosf(phi);
        float sphi = __sinf(phi);
        float R[6] = {};
        
        for (int t = 0; t<nproj; t++)
        {
            theta0 = theta[t];            
            float ctheta = __cosf(theta0);
            float stheta = __sinf(theta0);
            R[0] =  ctheta;       R[1] =  stheta;        R[2] = 0;
            R[3] =  stheta*cphi;  R[4] = -ctheta*cphi;   R[5] = sphi;
            u = R[0]*(tx-n/2)+R[1]*(ty-n/2)+n/2-sh[tz];
            v = R[3]*(tx-n/2)+R[4]*(ty-n/2)+R[5]*(sz-nz/2) + nz/2;
            
            ur = (int)(u-1e-5f);
            vr = (int)(v-1e-5f);            
            
            // linear interp            
            if ((ur >= 0) & (ur < n - 1) & (vr >= 0) & (vr < nz-1))
            {
                u = u-ur;
                v = v-vr;                
                f0 +=   data[ur+0+t*n+(vr+0)*n*nproj]*(1-u)*(1-v)+
                        data[ur+1+t*n+(vr+0)*n*nproj]*(0+u)*(1-v)+
                        data[ur+0+t*n+(vr+1)*n*nproj]*(1-u)*(0+v)+
                        data[ur+1+t*n+(vr+1)*n*nproj]*(0+u)*(0+v);
                        
            }
        }
        f[tx + (n-ty-1) * n + tz * n * n] += f0*c;        
    }  

    void __global__ backprojection_try_lamino(float *f, float *data, float *theta, float* phi, int sz, float c, int ncz, int n, int nz, int nproj)    
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= n || tz >= ncz)
            return;
        float u = 0;
        float v = 0;
        int ur = 0;
        int vr = 0;        
        
        float f0 = 0;
        float theta0 = 0;
        float cphi = __cosf(phi[tz]);
        float sphi = __sinf(phi[tz]);
        float R[6] = {};
        
        for (int t = 0; t<nproj; t++)
        {
            theta0 = theta[t];            
            float ctheta = __cosf(theta0);
            float stheta = __sinf(theta0);
            R[0] =  ctheta;       R[1] =  stheta;        R[2] = 0;
            R[3] =  stheta*cphi;  R[4] = -ctheta*cphi;   R[5] = sphi;
            u = R[0]*(tx-n/2)+R[1]*(ty-n/2)+n/2;
            v = R[3]*(tx-n/2)+R[4]*(ty-n/2)+R[5]*(sz-nz/2) + nz/2;
            
            ur = (int)(u-1e-5f);
            vr = (int)(v-1e-5f);                                    
            
            // linear interp            
            if ((ur >= 0) & (ur < n - 1) & (vr >= 0) & (vr < nz - 1))
            {
                u = u-ur;
                v = v-vr;                
                f0 +=   data[ur+0+t*n+(vr+0)*n*nproj]*(1-u)*(1-v)+
                        data[ur+1+t*n+(vr+0)*n*nproj]*(0+u)*(1-v)+
                        data[ur+0+t*n+(vr+1)*n*nproj]*(1-u)*(0+v)+
                        data[ur+1+t*n+(vr+1)*n*nproj]*(0+u)*(0+v);
                        
            }
        }
        f[tx + (n-ty-1) * n + tz * n * n] += f0*c;        
    }    
}
"""

module = cp.RawModule(code=source)
backprojection_kernel = module.get_function('backprojection')
backprojection_try_kernel = module.get_function('backprojection_try')
backprojection_try_lamino_kernel = module.get_function('backprojection_try_lamino')


class LineSummation():
    """Backprojection by summation over lines"""

    def __init__(self, theta, nproj, ncproj, nz, ncz, n, dtype):
        self.nproj = nproj
        self.ncproj = ncproj
        self.nz = nz
        self.ncz = ncz
        self.n = n
        self.theta = theta
        self.dtype = dtype

    def backprojection(self, f, data, stream=0, theta=[], lamino_angle=0, sz=0):
        if len(theta)==0:
            theta = self.theta
            f[:]=0
        f0 = f.astype('float32', copy=False)  # TODO: implement for float16
        data = data.astype('float32', copy=False)
        phi = cp.float32(cp.pi/2+(lamino_angle)/180*cp.pi)
        backprojection_kernel((int(cp.ceil(self.n/32)), int(cp.ceil(self.n/32+0.5)), self.ncz), (32, 32, 1),
                              (f0, data, theta, phi, sz, cp.float32(4.0/self.nproj), self.ncz, self.n, self.nz, self.ncproj))
        f[:] = f0.astype(self.dtype, copy=False)

    def backprojection_try(self, f, data, sh, stream=0, theta=[], lamino_angle=0, sz=0):
        if len(theta)==0:
            theta = self.theta
        f0 = f.astype('float32', copy=False)  # TODO: implement for float16
        data = data.astype('float32', copy=False)
        theta = theta.astype('float32', copy=False)
        phi = cp.float32(cp.pi/2+(lamino_angle)/180*cp.pi)
        backprojection_try_kernel((int(cp.ceil(self.n/32)), int(cp.ceil(self.n/32+0.5)), self.ncz), (32, 32, 1),
                                  (f0, data, theta, phi, sz, sh, cp.float32(4.0/self.nproj), self.ncz, self.n, self.nz, self.ncproj))
        f[:] = f0.astype(self.dtype, copy=False)

    def backprojection_try_lamino(self, f, data, sh, stream=0, theta=[], lamino_angle=0, sz=0):
        if len(theta)==0:
            theta = self.theta
        f0 = f.astype('float32', copy=False)  # TODO: implement for float16
        data = data.astype('float32', copy=False)
        # init lamino angle
        phi = (cp.pi/2+(lamino_angle+sh)/180*cp.pi).astype('float32')
        backprojection_try_lamino_kernel((int(cp.ceil(self.n/32)), int(cp.ceil(self.n/32+0.5)), self.ncz), (32, 32, 1),
                                         (f0, data, theta, phi, sz, cp.float32(4.0/self.nproj), self.ncz, self.n, self.nz, self.ncproj))
        f[:] = f.astype(self.dtype, copy=False)
