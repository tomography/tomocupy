#include "cfunc_linerec.cuh"
#include "kernels_linerec.cuh"

cfunc_linerec::cfunc_linerec(size_t nproj, size_t nz, size_t n,size_t ncproj, size_t ncz)
    : nproj(nproj), nz(nz), n(n), ncproj(ncproj), ncz(ncz) {    
  }


// destructor, memory deallocation
cfunc_linerec::~cfunc_linerec() { free(); }

void cfunc_linerec::free() {
  if (!is_free) {  
    is_free = true;   
  }
}

void cfunc_linerec::backprojection(size_t f_, size_t g_, size_t theta_, float phi, int sz, size_t stream_) {
    real* g = (real *)g_;    
    real* f = (real *)f_;
    float* theta = (float *)theta_;
    cudaStream_t stream = (cudaStream_t)stream_;        
    // set thread block, grid sizes will be computed before cuda kernel execution
    dim3 dimBlock(32,32,1);    
    dim3 GS3d0;  
    GS3d0 = dim3(ceil(n / 32.0), ceil(n / 32.0), ncz);
    backprojection_ker <<<GS3d0, dimBlock, 0, stream>>> (f, g, theta, phi, 4.0f/nproj, sz, ncz, n, nz, ncproj);
}                                            

void cfunc_linerec::backprojection_try(size_t f_, size_t g_, size_t theta_, size_t sh_, float phi, int sz,  size_t stream_) {
    real* g = (real *)g_;    
    real* f = (real *)f_;
    float* sh = (float *)sh_;

    float* theta = (float *)theta_;
    cudaStream_t stream = (cudaStream_t)stream_;        
    // set thread block, grid sizes will be computed before cuda kernel execution
    dim3 dimBlock(32,32,1);    
    dim3 GS3d0;  
    GS3d0 = dim3(ceil(n / 32.0), ceil(n / 32.0), ncz);
    backprojection_try_ker<<<GS3d0, dimBlock, 0, stream>>> (f, g, theta, phi, 4.0f/nproj, sz, sh, ncz, n, nz, ncproj);
}                                            

void cfunc_linerec::backprojection_try_lamino(size_t f_, size_t g_, size_t theta_, size_t phi_, int sz,  size_t stream_) {
    real* g = (real *)g_;    
    real* f = (real *)f_;
    float* phi = (float *)phi_;
    float* theta = (float *)theta_;
    cudaStream_t stream = (cudaStream_t)stream_;        
    // set thread block, grid sizes will be computed before cuda kernel execution
    dim3 dimBlock(32,32,1);    
    dim3 GS3d0;  
    GS3d0 = dim3(ceil(n / 32.0), ceil(n / 32.0), ncz);
    backprojection_try_lamino_ker<<<GS3d0, dimBlock, 0, stream>>> (f, g, theta, phi, 4.0f/nproj, sz, ncz, n, nz, ncproj);
}                                            