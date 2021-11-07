#include "cfunc_fourierrec.cuh"
#include "kernels_fourierrec.cuh"
#include "defs.cuh"
#include<stdio.h>
cfunc_fourierrec::cfunc_fourierrec(size_t ntheta, size_t pnz, size_t n, size_t theta_)
    : ntheta(ntheta), pnz(pnz), n(n) {
    float eps = 1e-2;
    mu = -log(eps) / (2 * n * n);
    m = ceil(2 * n * 1 / PI * sqrt(-mu * log(eps) + (mu * n) * (mu * n) / 4));    
    cudaMalloc((void **)&fdee,
            (2 * n + 2 * m) * (2 * n + 2 * m) * pnz * sizeof(float2));
    cudaMalloc((void **)&x, n * ntheta * sizeof(float));
    cudaMalloc((void **)&y, n * ntheta * sizeof(float));
    
    long long ffts[] = {2*n,2*n};
	long long idist = (2 * n + 2 * m) * (2 * n + 2 * m);long long odist = (2 * n + 2 * m) * (2 * n + 2 * m);
    long long inembed[] = {2 * n + 2 * m, 2 * n + 2 * m};long long onembed[] = {2 * n + 2 * m, 2 * n + 2 * m};
    size_t workSize = 0;

    cufftCreate(&plan2d);
    cufftXtMakePlanMany(plan2d, 
        2, ffts, 
        inembed, 1, idist, CUDA_C, 
        onembed, 1, odist, CUDA_C, 
        pnz, &workSize, CUDA_C);    
    // fft 1d
    cufftCreate(&plan1d);
    ffts[0] = n;
    idist = n;
    odist = n;
    inembed[0] = n;
    onembed[0] = n;
    cufftXtMakePlanMany(plan1d, 
        1, ffts, 
        inembed, 1, idist, CUDA_C, 
        onembed, 1, odist, CUDA_C, 
        ntheta*pnz, &workSize, CUDA_C);                   

    theta = (float*)theta_;

  }


// destructor, memory deallocation
cfunc_fourierrec::~cfunc_fourierrec() { free(); }

void cfunc_fourierrec::free() {
  if (!is_free) {
    cudaFree(fdee);
    cudaFree(x);
    cudaFree(y);
    cufftDestroy(plan2d);
    cufftDestroy(plan1d);
    is_free = true;   
  }
}

void cfunc_fourierrec::backprojection(size_t f_, size_t g_, size_t stream_) {
    float2* g = (float2 *)g_;    
    float2* f = (float2 *)f_;
    cudaStream_t stream = (cudaStream_t)stream_;    
    cufftSetStream(plan1d, stream);
    cufftSetStream(plan2d, stream);    

    // set thread block, grid sizes will be computed before cuda kernel execution
    dim3 dimBlock(BS1,BS2,BS3);    
    dim3 GS2d0,GS3d0,GS3d1,GS3d2,GS3d3;  
    GS2d0 = dim3(ceil(n / (float)BS1), ceil(ntheta / (float)BS2));
    GS3d0 = dim3(ceil(n / (float)BS1), ceil(n / (float)BS2),ceil(pnz / (float)BS3));
    GS3d1 = dim3(ceil(2 * n / (float)BS1), ceil(2 * n / (float)BS2),ceil(pnz / (float)BS3));
    GS3d2 = dim3(ceil((2 * n + 2 * m) / (float)BS1),ceil((2 * n + 2 * m) / (float)BS2), ceil(pnz / (float)BS3));
    GS3d3 = dim3(ceil(n / (float)BS1), ceil(ntheta / (float)BS2),ceil(pnz / (float)BS3));
   
    
    cudaMemsetAsync(fdee, 0, (2 * n + 2 * m) * (2 * n + 2 * m) * pnz * sizeof(float2),stream);

    takexy <<<GS2d0, dimBlock, 0, stream>>> (x, y, theta, n, ntheta);

    ifftshiftc <<<GS3d3, dimBlock, 0, stream>>> (g, n, ntheta, pnz);
    cufftXtExec(plan1d, (cufftComplex *)g, (cufftComplex *)g, CUFFT_FORWARD);
    ifftshiftc <<<GS3d3, dimBlock, 0, stream>>> (g, n, ntheta, pnz);
    gather <<<GS3d3, dimBlock, 0, stream>>> (g, fdee, x, y, m, mu, n, ntheta, pnz, TOMO_ADJ);
    

    wrap <<<GS3d2, dimBlock, 0, stream>>> (fdee, n, pnz, m, TOMO_ADJ);

    fftshiftc <<<GS3d2, dimBlock, 0, stream>>> (fdee, 2 * n + 2 * m, pnz);
    cufftXtExec(plan2d, (cufftComplex *)&fdee[m + m * (2 * n + 2 * m)],
                (cufftComplex *)&fdee[m + m * (2 * n + 2 * m)], CUFFT_INVERSE);
    fftshiftc <<<GS3d2, dimBlock, 0, stream>>> (fdee, 2 * n + 2 * m, pnz);
    
    divphi <<<GS3d0, dimBlock, 0, stream>>> (fdee, f, mu, n, pnz, m, TOMO_ADJ);    
}