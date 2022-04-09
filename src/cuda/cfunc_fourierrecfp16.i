/*interface*/
%module cfunc_fourierrecfp16

%{
#define SWIG_FILE_WITH_INIT
#include "cfunc_fourierrec.cuh"
%}

class cfunc_fourierrec
{
public:
  %immutable;
  size_t n;
  size_t nproj;
  size_t nz;
  
  %mutable;
  cfunc_fourierrec(size_t nproj, size_t nz, size_t n, size_t theta_);
  ~cfunc_fourierrec();
  void backprojection(size_t f, size_t g, size_t stream);
  void filter(size_t g, size_t w, size_t stream);
};
