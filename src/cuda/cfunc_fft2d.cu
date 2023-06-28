#include "kernels_fft2d.cu"
#include "cfunc_fft2d.cuh"

cfunc_fft2d::cfunc_fft2d(size_t ntheta_, size_t detw_, size_t deth_) {

  ntheta = ntheta_;
  detw = detw_;
  deth = deth_;

  long long ffts[]={deth,detw};
  long long idist = deth*detw;
  long long odist = deth*(detw/2+1);
  long long inembed[] = {deth,detw};
  long long onembed[] = {deth,detw/2+1};

//  cufftPlanMany(&plan2dchunk, 2, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_R2C, ntheta);
  size_t workSize = 0;
  cufftCreate(&plan2dchunk);
  cufftXtMakePlanMany(plan2dchunk, 
      2, ffts, 
      inembed, 1, idist, CUDA_R_32F, 
      onembed, 1, odist, CUDA_C_32F, 
      ntheta, &workSize, CUDA_C_32F);  

  BS2d = dim3(32, 32, 1);
  GS2d0 = dim3(ceil(detw / (float)BS2d.x), ceil(deth / (float)BS2d.y), ceil(ntheta / (float)BS2d.z));
}

// destructor, memory deallocation
cfunc_fft2d::~cfunc_fft2d() { free(); }

void cfunc_fft2d::free() {
  if (!is_free) {
    cufftDestroy(plan2dchunk);
    is_free = true;
  }
}

void cfunc_fft2d::adj(size_t g_, size_t f_, size_t stream_) {

  /*f = (float *)f_;
  g = (float2 *)g_;
  stream = (cudaStream_t)stream_;    
  
  cufftSetStream(plan2dchunk, stream);
  rfftshiftc2d<<<GS2d0, BS2d, 0, stream>>>(f, detw, deth, ntheta);
  cufftExecR2C(plan2dchunk, (cufftReal *)f, (cufftComplex *)g);
  //irfftshiftc2d<<<GS2d0, BS2d, 0, stream>>>(g, detw/2+1, deth, ntheta);  
  mulc<<<GS2d0, BS2d, 0, stream>>>(g, detw/2+1, deth, ntheta, 1.f/(deth*detw));  */
}

void cfunc_fft2d::fwd(size_t g_, size_t f_, size_t stream_) {

  f = (float *)f_;
  g = (float2 *)g_;
  stream = (cudaStream_t)stream_;    
  
  cufftSetStream(plan2dchunk, stream);
  rfftshiftc2d<<<GS2d0, BS2d, 0, stream>>>(f, detw, deth, ntheta);
  cufftXtExec(plan2dchunk, f,g, CUFFT_FORWARD);        
  irfftshiftc2d<<<GS2d0, BS2d, 0, stream>>>(g, detw/2+1, deth, ntheta);  
  mulc<<<GS2d0, BS2d, 0, stream>>>(g, detw/2+1, deth, ntheta, 1.f/(deth*detw));  
}
