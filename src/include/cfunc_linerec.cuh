#ifndef CFUNC_LINEREC_CUH
#define CFUNC_LINEREC_CUH

#include <cufft.h>
#include <cuda_fp16.h>
#include "defs.cuh"


class cfunc_linerec {
  bool is_free = false;
  
public:
  size_t n;      // width of square slices
  size_t nproj; // number of angles
  size_t nz;    // number of slices
  size_t ncproj;    // number of slices
  size_t ncz;    // number of slices
  cfunc_linerec(size_t nproj, size_t nz, size_t n, size_t ncproj, size_t ncz);
  ~cfunc_linerec();
  void backprojection(size_t f_, size_t g_, size_t theta, float phi, int sz, size_t stream_);
  void backprojection_try(size_t f_, size_t g_, size_t theta_,size_t sh_,  float phi, int sz, size_t stream_);
  void backprojection_try_lamino(size_t f_, size_t g_, size_t theta_, size_t phi_, int sz,  size_t stream_);
  void free();
};

#endif