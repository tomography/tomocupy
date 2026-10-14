#ifndef CFUNC_FOURIERREC_CUH
#define CFUNC_FOURIERREC_CUH

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>
#include "defs.cuh"


class cfunc_fourierrec {
  bool is_free = false;

  size_t m;
  float mu;

  
  float *x;
  float *y;
  float* theta;
  real2 *fde;

  cufftHandle plan2d;
  cufftHandle plan1d;

  // pre-computed kernel grid dimensions (constant for lifetime of object)
  dim3 GS2d0, GS3d0, GS3d2, GS3d3;

  // binned-scatter index (see build_bins); one copy serves every z and every
  // call, since theta and the detector grid are fixed for this object
  int tile_b = 32;              // tile side on the padded grid
  int subsize = 2048;           // samples per subproblem
  int ntx = 0;                  // tiles per row
  int nsub = 0;                 // number of subproblems
  size_t shmem = 0;             // dynamic shared memory per block
  int *bin_samples = nullptr;   // sample id of every (sample, tile) pair
  int *sub_tile = nullptr;      // per subproblem: which tile,
  int *sub_beg = nullptr;       //   and which slice of that tile's run
  int *sub_end = nullptr;
  int *sub_scratch = nullptr;   // scratch slot of a split tile, else -1
  int nscratch = 0;             // number of split tiles
  int *fin_tile = nullptr;      // tile id of each scratch slot
  float2 *scratch = nullptr;    // float32 partial sums of the split tiles
  size_t scratch_bytes = 0;

  void build_bins();

public:
  size_t n;      // width of square slices
  size_t nproj; // number of angles
  size_t nz;    // number of slices
  cfunc_fourierrec(size_t nproj, size_t nz, size_t n, size_t theta);
  ~cfunc_fourierrec();
  void backprojection(size_t f, size_t g, size_t stream);
  void free();
};

#endif