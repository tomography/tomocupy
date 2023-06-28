#ifndef CFUNC_FFT2D_CUH
#define CFUNC_FFT2D_CUH

#include <cufft.h>
#include <cufftXt.h>


class cfunc_fft2d {
  bool is_free = false;
  
  float *f;
  float2 *g;
  cufftHandle plan2dchunk;
  cudaStream_t stream;

  dim3 BS2d, GS2d0;
  
  size_t ntheta,detw,deth;
  
public:  
  cfunc_fft2d(size_t ntheta, size_t detw, size_t deth);  
  ~cfunc_fft2d();  
  void fwd(size_t g_, size_t f_, size_t stream);
  void adj(size_t g_, size_t f_, size_t stream);
  void free();
};

#endif
