/*interface*/
%module cfunc_fourierrec

%{
#define SWIG_FILE_WITH_INIT
#include "cfunc_fourierrec.cuh"
%}

class cfunc_fourierrec
{
public:
  %immutable;
  size_t n;
  size_t ntheta;
  size_t pnz;
  
  %mutable;
  cfunc_fourierrec(size_t ntheta, size_t pnz, size_t n, size_t theta_);
  ~cfunc_fourierrec();
  void backprojection(size_t f, size_t g, size_t stream);
};
