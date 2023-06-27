#ifndef CFUNC_FFT_CUH
#define CFUNC_FFT_CUH

#include <cufft.h>


class cfunc_usfft2d {
  bool is_free = false;
  
  
  float2 *f;
  float2 *g;
  
  float2 *fdee2d;
  float* x;
  float* y;
  float* theta;
  cufftHandle plan2dchunk;
  cudaStream_t stream;
  
  dim3 BS2d, GS2d0, GS2d1, GS2d2;
  
  size_t n0,n1,n2;  
  size_t ntheta,detw,deth;
  size_t m0,m1;
  float mu0;float mu1;
public:  
  cfunc_usfft2d(size_t n0, size_t n1, size_t n2, size_t ntheta, size_t detw, size_t deth);  
  ~cfunc_usfft2d();  
  void fwd(size_t g_, size_t f_, size_t theta_, float phi, int k, int deth0, size_t stream_);
  void adj(size_t f_, size_t g_, size_t theta_, float phi, int k, int deth0, size_t stream_);
  void free();
};

#endif
