/*interface*/
%module cfunc_filter

%{
#define SWIG_FILE_WITH_INIT
#include "cfunc_filter.cuh"
%}

class cfunc_filter
{
public:
  %immutable;
  size_t n;
  size_t nproj;
  size_t nz;
  
  %mutable;
  cfunc_filter(size_t nproj, size_t nz, size_t n);
  ~cfunc_filter();
  void filter(size_t g, size_t w, size_t stream);
};
