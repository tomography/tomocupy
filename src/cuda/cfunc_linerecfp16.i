/*interface*/
%module cfunc_linerecfp16

%{
#define SWIG_FILE_WITH_INIT
#include "cfunc_linerec.cuh"
%}

class cfunc_linerec
{
public:
  %immutable;
  size_t n;      // width of square slices
  size_t nproj; // number of angles
  size_t nz;    // number of slices
  size_t ncproj;    // number of slices
  size_t ncz;    // number of slices

  %mutable;
  cfunc_linerec(size_t nproj, size_t nz, size_t n, size_t ncproj, size_t ncz);
  ~cfunc_linerec();
  void backprojection(size_t f_, size_t g_, size_t theta, float phi, int sz, size_t stream_);  
  void backprojection_try(size_t f_, size_t g_, size_t theta_,size_t sh_,  float phi, int sz, size_t stream_);
  void backprojection_try_lamino(size_t f_, size_t g_, size_t theta_, size_t phi_, int sz,  size_t stream_);
};
