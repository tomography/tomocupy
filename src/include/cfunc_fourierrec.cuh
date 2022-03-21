#ifndef CFUNC_FOURIERREC_CUH
#define CFUNC_FOURIERREC_CUH

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>


class cfunc_fourierrec {
  bool is_free = false;

  size_t m;
  float mu;

  
  float *x;
  float *y;
  float* theta;
  half2 *fde;
  half2 *ge;

  cufftHandle plan2d;  
  cufftHandle plan1d;
  cufftHandle plan_filter_fwd;
  cufftHandle plan_filter_inv;

public:
  size_t n;      // width of square slices
  size_t ntheta; // number of angles
  size_t pnz;    // number of slices
  size_t ne;
  cfunc_fourierrec(size_t ntheta, size_t pnz, size_t n, size_t theta);
  ~cfunc_fourierrec();
  void backprojection(size_t f, size_t g, size_t stream);
  void filter(size_t g, size_t w, size_t stream);
  void free();
};

#endif