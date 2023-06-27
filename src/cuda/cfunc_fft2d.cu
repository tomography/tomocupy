#include "kernels_fft2d.cu"
#include "cfunc_fft2d.cuh"

cfunc_fft2d::cfunc_fft2d(size_t ntheta_, size_t detw_, size_t deth_) {

  ntheta = ntheta_;
  detw = detw_;
  deth = deth_;

  int ffts[2];
  int idist;
  int inembed[2];

  // cfunc_fft2d 2d
  ffts[0] = deth;
  ffts[1] = detw;
  idist = deth*detw;
  inembed[0] = deth;
  inembed[1] = detw;

  cufftPlanMany(&plan2dchunk, 2, ffts, inembed, 1, idist, inembed, 1, idist, CUFFT_C2C, ntheta);

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

  f = (float2 *)f_;
  g = (float2 *)g_;
  stream = (cudaStream_t)stream_;    
  
  cufftSetStream(plan2dchunk, stream);
  fftshiftc2d<<<GS2d0, BS2d, 0, stream>>>(f, detw, deth, ntheta);
  cufftExecC2C(plan2dchunk, (cufftComplex *)f, (cufftComplex *)g, CUFFT_INVERSE);
  fftshiftc2d<<<GS2d0, BS2d, 0, stream>>>(g, detw, deth, ntheta);  
  mulc<<<GS2d0, BS2d, 0, stream>>>(g, detw, deth, ntheta, 1.f/(deth*detw));  
}

void cfunc_fft2d::fwd(size_t g_, size_t f_, size_t stream_) {

  f = (float2 *)f_;
  g = (float2 *)g_;
  stream = (cudaStream_t)stream_;    
  
  cufftSetStream(plan2dchunk, stream);
  fftshiftc2d<<<GS2d0, BS2d, 0, stream>>>(f, detw, deth, ntheta);
  cufftExecC2C(plan2dchunk, (cufftComplex *)f, (cufftComplex *)g, CUFFT_FORWARD);
  fftshiftc2d<<<GS2d0, BS2d, 0, stream>>>(g, detw, deth, ntheta);  
  mulc<<<GS2d0, BS2d, 0, stream>>>(g, detw, deth, ntheta, 1.f/(deth*detw));  
}
