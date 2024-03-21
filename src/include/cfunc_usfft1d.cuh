#ifndef CFUNC_FFT_CUH
#define CFUNC_FFT_CUH

#include <cufft.h>


class cfunc_usfft1d {
  bool is_free = false;
  
  
  float *f;
  float2 *g;
  
  float2 *fdee1d;
    
  float* x;
  cufftHandle plan1dchunk;
  cudaStream_t stream;
  
  dim3 BS1d, GS1d0, GS1d1, GS1d2;
  dim3 BS1dx, GS1dx;

  size_t n0,n1,n2;  
  size_t deth;  
  size_t m2;
  float mu2;
public:  
  
  cfunc_usfft1d(size_t n0, size_t n1, size_t n2, size_t deth);
  ~cfunc_usfft1d();  
  void fwd(size_t g_, size_t f_, float phi, size_t stream_);  
  void adj(size_t g_, size_t f_, float phi, size_t stream_);  
  void free();
};

#endif
