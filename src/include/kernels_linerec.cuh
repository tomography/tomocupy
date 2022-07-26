#include "defs.cuh"

void __global__ backprojection_ker(real *f, real *data, float *theta, float phi, float c, int sz, int ncz, int n, int nz, int nproj)
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
    
    real f0 = 0;
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
            f0 +=   data[ur+0+t*n+(vr+0)*n*nproj]*static_cast<real>((1-u)*(1-v))+
                    data[ur+1+t*n+(vr+0)*n*nproj]*static_cast<real>((0+u)*(1-v))+
                    data[ur+0+t*n+(vr+1)*n*nproj]*static_cast<real>((1-u)*(0+v))+
                    data[ur+1+t*n+(vr+1)*n*nproj]*static_cast<real>((0+u)*(0+v));
        }
    }
    f[tx + (n-ty-1) * n + tz * n * n] += static_cast<real>((float)f0*c);        
}    

void __global__ backprojection_try_ker(real *f, real *data, float *theta, float phi, float c, int sz, float* sh, int ncz, int n, int nz, int nproj)
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
    
    real f0 = 0;
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
            f0 +=   data[ur+0+t*n+(vr+0)*n*nproj]*static_cast<real>((1-u)*(1-v))+
                    data[ur+1+t*n+(vr+0)*n*nproj]*static_cast<real>((0+u)*(1-v))+
                    data[ur+0+t*n+(vr+1)*n*nproj]*static_cast<real>((1-u)*(0+v))+
                    data[ur+1+t*n+(vr+1)*n*nproj]*static_cast<real>((0+u)*(0+v));
                    
        }
    }
    f[tx + (n-ty-1) * n + tz * n * n] += static_cast<real>((float)f0*c);        
}  
void __global__ backprojection_try_lamino_ker(real *f, real *data, float *theta, float* phi, float c, int sz, int ncz, int n, int nz, int nproj)    
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
    
    real f0 = 0;
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
            f0 +=   data[ur+0+t*n+(vr+0)*n*nproj]*static_cast<real>((1-u)*(1-v))+
                    data[ur+1+t*n+(vr+0)*n*nproj]*static_cast<real>((0+u)*(1-v))+
                    data[ur+0+t*n+(vr+1)*n*nproj]*static_cast<real>((1-u)*(0+v))+
                    data[ur+1+t*n+(vr+1)*n*nproj]*static_cast<real>((0+u)*(0+v));
                    
        }
    }
    f[tx + (n-ty-1) * n + tz * n * n] += static_cast<real>((float)f0*c);          
}    
