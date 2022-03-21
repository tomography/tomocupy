#include "cfunc_fourierrec.cuh"
#include "kernels_fourierrec.cuh"
#include<stdio.h>
cfunc_fourierrec::cfunc_fourierrec(size_t ntheta, size_t pnz, size_t n, size_t theta_)
    : ntheta(ntheta), pnz(pnz), n(n) {
    float eps = 1e-2;
    mu = -log(eps) / (2 * n * n);
    ne = pow(2,ceil(log2(3*n/2)));
    m = ceil(2 * n * 1 / PI * sqrt(-mu * log(eps) + (mu * n) * (mu * n) / 4));    
    cudaMalloc((void **)&fde,
            (2 * n + 2 * m) * (2 * n + 2 * m) * pnz * sizeof(half2));
    cudaMalloc((void **)&ge,
            (ne/2+1) * ntheta * 2 * pnz * sizeof(half2));
    cudaMalloc((void **)&x, n * ntheta * sizeof(float));
    cudaMalloc((void **)&y, n * ntheta * sizeof(float));
    
    
    long long ffts[] = {2*n,2*n};
	  long long idist = (2 * n + 2 * m) * (2 * n + 2 * m);long long odist = (2 * n + 2 * m) * (2 * n + 2 * m);
    long long inembed[] = {2 * n + 2 * m, 2 * n + 2 * m};long long onembed[] = {2 * n + 2 * m, 2 * n + 2 * m};
    size_t workSize = 0;

    cufftCreate(&plan2d);
    cufftXtMakePlanMany(plan2d, 
        2, ffts, 
        inembed, 1, idist, CUDA_C_16F, 
        onembed, 1, odist, CUDA_C_16F, 
        pnz, &workSize, CUDA_C_16F);    
    // fft 1d
    cufftCreate(&plan1d);
    ffts[0] = n;
    idist = n;
    odist = n;
    inembed[0] = n;
    onembed[0] = n;
    cufftXtMakePlanMany(plan1d, 
        1, ffts, 
        inembed, 1, idist, CUDA_C_16F, 
        onembed, 1, odist, CUDA_C_16F, 
        ntheta*pnz, &workSize, CUDA_C_16F);                   

    //fft filter R<->C
    cufftCreate(&plan_filter_fwd);
    cufftCreate(&plan_filter_inv);
    
    ffts[0] = ne;
	  idist = ne;odist = ne/2+1;
    inembed[0] = ne;onembed[0] = ne/2+1;
    cufftXtMakePlanMany(plan_filter_fwd, 
        1, ffts, 
        inembed, 1, idist, CUDA_R_16F, 
        onembed, 1, odist, CUDA_C_16F, 
        2*ntheta*pnz, &workSize, CUDA_C_16F);      
    cufftXtMakePlanMany(plan_filter_inv, 
        1, ffts, 
        onembed, 1, odist, CUDA_C_16F, 
        inembed, 1, idist, CUDA_R_16F, 
        2*ntheta*pnz, &workSize, CUDA_C_16F);
    
    theta = (float*)theta_;
  }


// destructor, memory deallocation
cfunc_fourierrec::~cfunc_fourierrec() { free(); }

void cfunc_fourierrec::free() {
  if (!is_free) {
    cudaFree(fde);
    cudaFree(ge);
    cudaFree(x);
    cudaFree(y);
    cufftDestroy(plan2d);
    cufftDestroy(plan1d);
    cufftDestroy(plan_filter_fwd);
    cufftDestroy(plan_filter_inv);
    is_free = true;   
  }
}

void cfunc_fourierrec::backprojection(size_t f_, size_t g_, size_t stream_) {
    half2* g = (half2 *)g_;    
    half2* f = (half2 *)f_;
    cudaStream_t stream = (cudaStream_t)stream_;    
    cufftSetStream(plan1d, stream);
    cufftSetStream(plan2d, stream);    

    // set thread block, grid sizes will be computed before cuda kernel execution
    dim3 dimBlock(32,32,1);    
    dim3 GS2d0,GS3d0,GS3d1,GS3d2,GS3d3;  
    GS2d0 = dim3(ceil(n / 32.0), ceil(ntheta / 32.0));
    GS3d0 = dim3(ceil(n / 32.0), ceil(n / 32.0),pnz);
    GS3d1 = dim3(ceil(2 * n / 32.0), ceil(2 * n / 32.0),pnz);
    GS3d2 = dim3(ceil((2 * n + 2 * m) / 32.0),ceil((2 * n + 2 * m) / 32.0), pnz);
    GS3d3 = dim3(ceil(n / 32.0), ceil(ntheta / 32.0),pnz);
   
    
    cudaMemsetAsync(fde, 0, (2 * n + 2 * m) * (2 * n + 2 * m) * pnz * sizeof(half2),stream);

    takexy <<<GS2d0, dimBlock, 0, stream>>> (x, y, theta, n, ntheta);


    mulc <<<GS3d3, dimBlock, 0, stream>>> (g, 1/(float)n, n, ntheta, pnz);
    ifftshiftc <<<GS3d3, dimBlock, 0, stream>>> (g, n, ntheta, pnz);
    cufftXtExec(plan1d, g, g, CUFFT_FORWARD);
    ifftshiftc <<<GS3d3, dimBlock, 0, stream>>> (g, n, ntheta, pnz);    
    
    gather <<<GS3d3, dimBlock, 0, stream>>> (g, fde, x, y, m, mu, n, ntheta, pnz);    
    wrap <<<GS3d2, dimBlock, 0, stream>>> (fde, n, pnz, m);
    
    fftshiftc <<<GS3d2, dimBlock, 0, stream>>> (fde, 2 * n + 2 * m, pnz);
    cufftXtExec(plan2d, &fde[m + m * (2 * n + 2 * m)],
               &fde[m + m * (2 * n + 2 * m)], CUFFT_INVERSE);
    fftshiftc <<<GS3d2, dimBlock, 0, stream>>> (fde, 2 * n + 2 * m, pnz);
    
    divphi <<<GS3d0, dimBlock, 0, stream>>> (fde, f, mu, n, pnz, ntheta, m);    
    circ <<<GS3d0, dimBlock,0,stream>>> (f, 1.0f / n, n, pnz);    
}

void cfunc_fourierrec::filter(size_t g_, size_t w_, size_t stream_) {
    half* g = (half *)g_;    
    half2* w = (half2 *)w_;
    cudaStream_t stream = (cudaStream_t)stream_;    
    cufftSetStream(plan_filter_fwd, stream);
    cufftSetStream(plan_filter_inv, stream);    
    dim3 dimBlock(32,32,1);        
    dim3 GS3d1 = dim3(ceil(ne/32.0), ceil(ntheta / 32.0),2*pnz);
    dim3 GS3d2 = dim3(ceil((ne/2+1)/32.0), ceil(ntheta / 32.0),2*pnz);
    cufftXtExec(plan_filter_fwd, g, ge, CUFFT_FORWARD);
    mulw <<<GS3d2, dimBlock, 0, stream>>> (ge, w, ne/2+1, ntheta, 2*pnz);
    cufftXtExec(plan_filter_inv, ge, g, CUFFT_INVERSE);
}
