"""
CUDA Raw kernels for computing back-projection
"""

import cupy as cp

source = """
extern "C" {        
    void __global__ adj(float *f, float *data, float *theta, float phi, int sz, int ncz, int n, int nz, int nproj)
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
                f0 +=   data[ur+0+(vr+0)*n+t*n*nz]*(1-u)*(1-v)+
                        data[ur+1+(vr+0)*n+t*n*nz]*(0+u)*(1-v)+
                        data[ur+0+(vr+1)*n+t*n*nz]*(1-u)*(0+v)+
                        data[ur+1+(vr+1)*n+t*n*nz]*(0+u)*(0+v);
                        
            }
        }
        f[tx + ty * n + tz * n * n] += f0;        
    }    

    void __global__ adj_try(float *f, float *data, float *theta, float phi, int sz, float* sh, int nsh, int n, int nz, int nproj)
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= n || tz >= nsh)
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
                f0 +=   data[ur+0+(vr+0)*n+t*n*nz]*(1-u)*(1-v)+
                        data[ur+1+(vr+0)*n+t*n*nz]*(0+u)*(1-v)+
                        data[ur+0+(vr+1)*n+t*n*nz]*(1-u)*(0+v)+
                        data[ur+1+(vr+1)*n+t*n*nz]*(0+u)*(0+v);
                        
            }
        }
        f[tx + ty * n + tz * n * n] += f0;        
    }  

    void __global__ adj_try_lamino(float *f, float *data, float *theta, float* phi, int sz, int nsh, int n, int nz, int nproj)    
    {
        int tx = blockDim.x * blockIdx.x + threadIdx.x;
        int ty = blockDim.y * blockIdx.y + threadIdx.y;
        int tz = blockDim.z * blockIdx.z + threadIdx.z;
        if (tx >= n || ty >= n || tz >= nsh)
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
                f0 +=   data[ur+0+(vr+0)*n+t*n*nz]*(1-u)*(1-v)+
                        data[ur+1+(vr+0)*n+t*n*nz]*(0+u)*(1-v)+
                        data[ur+0+(vr+1)*n+t*n*nz]*(1-u)*(0+v)+
                        data[ur+1+(vr+1)*n+t*n*nz]*(0+u)*(0+v);
                        
            }
        }
        f[tx + ty * n + tz * n * n] += f0;        
    }    
}
"""

module = cp.RawModule(code=source)
adj_kernel = module.get_function('adj')
adj_try_kernel = module.get_function('adj_try')
adj_try_lamino_kernel = module.get_function('adj_try_lamino')


def adj(f, data, theta, lamino_angle, sz):
    [ncz, n] = f.shape[:2]
    [nproj, nz] = data.shape[:2]
    phi = cp.float32(cp.pi/2+(lamino_angle)/180*cp.pi)

    adj_kernel((int(cp.ceil(n/32)), int(cp.ceil(n/32+0.5)), ncz), (32, 32, 1),
               (f, data, theta, phi, sz, ncz, n, nz, nproj))


def adj_try(f, data, theta, lamino_angle, sz, sh):
    [nsh, n] = f.shape[:2]
    [nproj, nz] = data.shape[:2]
    phi = cp.float32(cp.pi/2+(lamino_angle)/180*cp.pi)
    adj_try_kernel((int(cp.ceil(n/32)), int(cp.ceil(n/32+0.5)), nsh), (32, 32, 1),
                   (f, data, theta, phi, sz, sh, nsh, n, nz, nproj))


def adj_try_lamino(f, data, theta, lamino_angle, sz, sh):
    [nsh, n] = f.shape[:2]
    [nproj, nz] = data.shape[:2]

    # init lamino angle
    phi = (cp.pi/2+(lamino_angle+sh)/180*cp.pi).astype('float32')
    adj_try_lamino_kernel((int(cp.ceil(n/32)), int(cp.ceil(n/32+0.5)), nsh), (32, 32, 1),
                          (f, data, theta, phi, sz, nsh, n, nz, nproj))
