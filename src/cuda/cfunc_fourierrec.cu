#include "cfunc_fourierrec.cuh"
#include "kernels_fourierrec.cuh"

cfunc_fourierrec::cfunc_fourierrec(size_t nproj, size_t nz, size_t n, size_t theta_)
    : nproj(nproj), nz(nz), n(n) {
    float eps = 1e-3;
    mu = -log(eps) / (2 * n * n);        
    m = ceil(2 * n * 1 / PI * sqrt(-mu * log(eps) + (mu * n) * (mu * n) / 4));    
    cudaMalloc((void **)&fde,
            (2 * n + 2 * m) * (2 * n + 2 * m) * nz * sizeof(real2));
    cudaMalloc((void **)&x, n * nproj * sizeof(float));
    cudaMalloc((void **)&y, n * nproj * sizeof(float));
    
    long long ffts[] = {2*n,2*n};
	  long long idist = (2 * n + 2 * m) * (2 * n + 2 * m);long long odist = (2 * n + 2 * m) * (2 * n + 2 * m);
    long long inembed[] = {2 * n + 2 * m, 2 * n + 2 * m};long long onembed[] = {2 * n + 2 * m, 2 * n + 2 * m};
    size_t workSize = 0;

    cufftCreate(&plan2d);
    cufftXtMakePlanMany(plan2d, 
        2, ffts, 
        inembed, 1, idist, CUDA_C, 
        onembed, 1, odist, CUDA_C, 
        nz, &workSize, CUDA_C);    
    // fft 1d
    cufftCreate(&plan1d);
    ffts[0] = n;
    idist = n;
    odist = n;
    inembed[0] = n;
    onembed[0] = n;
    cufftXtMakePlanMany(plan1d, 
        1, ffts, 
        inembed, 1, idist, CUDA_C, 
        onembed, 1, odist, CUDA_C, 
        nproj*nz, &workSize, CUDA_C);                      
    
    theta = (float*)theta_;

    // Pre-compute grid dimensions (constant for lifetime of object)
    dim3 dimBlock(32, 32, 1);
    GS2d0 = dim3(ceil(n / 32.0), ceil(nproj / 32.0));
    GS3d0 = dim3(ceil(n / 32.0), ceil(n / 32.0), nz);
    GS3d2 = dim3(ceil((2 * n + 2 * m) / 32.0), ceil((2 * n + 2 * m) / 32.0), nz);
    GS3d3 = dim3(ceil(n / 32.0), ceil(nproj / 32.0), nz);

    // Pre-compute x, y once (theta is fixed for lifetime of object)
    takexy <<<GS2d0, dimBlock>>> (x, y, theta, n, nproj);

    build_bins();
  }


// Which samples reach which tile of the padded spectrum, for scatter_binned.
// Built once: theta and the detector grid are fixed and every z shares it.
// A sample is listed under each of the <=4 tiles its stencil touches
// (duplication ~1.6 at b=32); tiles over `subsize` samples are split.
void cfunc_fourierrec::build_bins() {
  const size_t s = 2 * n + 2 * m;
  if ((size_t)tile_b < 2 * m + 1)
    tile_b = 2 * m + 1;
  ntx = (int)((s + tile_b - 1) / tile_b);
  const int ntile = ntx * ntx;

  int *counts;
  cudaMalloc((void **)&counts, (ntile + 1) * sizeof(int));
  cudaMemset(counts, 0, (ntile + 1) * sizeof(int));

  dim3 dimBlock(32, 32, 1);
  count_tiles <<<GS2d0, dimBlock>>> (counts, x, y, m, n, nproj, tile_b, ntx);

  int *hcounts = new int[ntile];
  cudaMemcpy(hcounts, counts, ntile * sizeof(int), cudaMemcpyDeviceToHost);

  int *hoffsets = new int[ntile + 1];
  hoffsets[0] = 0;
  for (int i = 0; i < ntile; i++)
    hoffsets[i + 1] = hoffsets[i] + hcounts[i];
  const int total = hoffsets[ntile];

  int *offsets;
  cudaMalloc((void **)&offsets, (ntile + 1) * sizeof(int));
  cudaMemcpy(offsets, hoffsets, (ntile + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemset(counts, 0, ntile * sizeof(int));   // reuse as the fill cursor
  cudaMalloc((void **)&bin_samples, (size_t)total * sizeof(int));
  fill_tiles <<<GS2d0, dimBlock>>> (bin_samples, counts, offsets, x, y, m, n,
                                    nproj, tile_b, ntx);
  cudaFree(counts);
  cudaFree(offsets);

  // empty tiles get no subproblem; fde is already zeroed
  nsub = 0;
  nscratch = 0;
  for (int i = 0; i < ntile; i++) {
    const int k = (hcounts[i] + subsize - 1) / subsize;
    nsub += k;
    if (k > 1)
      nscratch++;
  }

  int *htile = new int[nsub];
  int *hbeg = new int[nsub];
  int *hend = new int[nsub];
  int *hslot = new int[nsub];
  int *hfin = new int[nscratch > 0 ? nscratch : 1];
  for (int i = 0, p = 0, q = 0; i < ntile; i++) {
    const int k = (hcounts[i] + subsize - 1) / subsize;
    if (k > 1)
      hfin[q] = i;
    for (int j = 0; j < k; j++, p++) {
      const int e = hoffsets[i] + (j + 1) * subsize;
      htile[p] = i;
      hbeg[p] = hoffsets[i] + j * subsize;
      hend[p] = e < hoffsets[i + 1] ? e : hoffsets[i + 1];
      hslot[p] = k > 1 ? q : -1;
    }
    if (k > 1)
      q++;
  }

  cudaMalloc((void **)&sub_tile, nsub * sizeof(int));
  cudaMalloc((void **)&sub_beg, nsub * sizeof(int));
  cudaMalloc((void **)&sub_end, nsub * sizeof(int));
  cudaMalloc((void **)&sub_scratch, nsub * sizeof(int));
  cudaMemcpy(sub_tile, htile, nsub * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(sub_beg, hbeg, nsub * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(sub_end, hend, nsub * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(sub_scratch, hslot, nsub * sizeof(int), cudaMemcpyHostToDevice);

#ifdef HALF
  // only the half build needs the float32 landing ground
  if (nscratch > 0) {
    scratch_bytes = (size_t)nscratch * nz * tile_b * tile_b * sizeof(float2);
    cudaMalloc((void **)&fin_tile, nscratch * sizeof(int));
    cudaMemcpy(fin_tile, hfin, nscratch * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&scratch, scratch_bytes);
  }
#endif

  delete[] htile;
  delete[] hbeg;
  delete[] hend;
  delete[] hslot;
  delete[] hfin;
  delete[] hcounts;
  delete[] hoffsets;

  shmem = 2 * (size_t)tile_b * (tile_b + 1) * sizeof(float);
}


// destructor, memory deallocation
cfunc_fourierrec::~cfunc_fourierrec() { free(); }

void cfunc_fourierrec::free() {
  if (!is_free) {
    cudaFree(fde);
    cudaFree(x);
    cudaFree(y);
    cudaFree(bin_samples);
    cudaFree(sub_tile);
    cudaFree(sub_beg);
    cudaFree(sub_end);
    cudaFree(sub_scratch);
    cudaFree(fin_tile);
    cudaFree(scratch);
    cufftDestroy(plan2d);
    cufftDestroy(plan1d);
    is_free = true;
  }
}

void cfunc_fourierrec::backprojection(size_t f_, size_t g_, size_t stream_) {
    real2* g = (real2 *)g_;    
    real2* f = (real2 *)f_;
    cudaStream_t stream = (cudaStream_t)stream_;    
    cufftSetStream(plan1d, stream);
    cufftSetStream(plan2d, stream);    

    dim3 dimBlock(32, 32, 1);

    cudaMemsetAsync(fde, 0, (2 * n + 2 * m) * (2 * n + 2 * m) * nz * sizeof(real2), stream);

    ifftshiftc <<<GS3d3, dimBlock, 0, stream>>> (g, n, nproj, nz);
    cufftXtExec(plan1d, g, g, CUFFT_FORWARD);
    ifftshiftc <<<GS3d3, dimBlock, 0, stream>>> (g, n, nproj, nz);    
    
    if (scratch)
      cudaMemsetAsync(scratch, 0, scratch_bytes, stream);
    scatter_binned <<<dim3(nsub, nz), SCATTER_CH, shmem, stream>>> (
        fde, g, x, y, bin_samples, sub_tile, sub_beg, sub_end, sub_scratch,
        scratch, m, mu, n, nproj, nz, tile_b, ntx, 2 * n + 2 * m);
    if (scratch)
      finalize_tiles <<<dim3(nscratch, nz), SCATTER_CH, 0, stream>>> (
          fde, scratch, fin_tile, nz, tile_b, ntx, 2 * n + 2 * m);

    wrap <<<GS3d2, dimBlock, 0, stream>>> (fde, n, nz, m);
    
    fftshiftc <<<GS3d2, dimBlock, 0, stream>>> (fde, 2 * n + 2 * m, nz);
    cufftXtExec(plan2d, &fde[m + m * (2 * n + 2 * m)],
               &fde[m + m * (2 * n + 2 * m)], CUFFT_INVERSE);
    fftshiftc <<<GS3d2, dimBlock, 0, stream>>> (fde, 2 * n + 2 * m, nz);
    
    divphi <<<GS3d0, dimBlock, 0, stream>>> (fde, f, mu, n, nz, nproj, m);        
    circ <<<GS3d0, dimBlock, 0, stream>>> (f, 0, n, nz);  

}
