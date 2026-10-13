#include "defs.cuh"
#define PI 3.1415926535f

void __global__ divphi(real2 *g, real2 *f, float mu, int n, int nz, int nproj, int m)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n || ty >= n || tz >= nz)
    return;
  float dx = tx / (float)n - 0.5;
  float dy = ty / (float)n - 0.5;
  //note overfilling with computing exp and float16 precision
  real phi = static_cast<real>(__expf(mu * (n * n) * (dx * dx + dy * dy)) / nproj);
  phi *= (1-n%4);////1-n%4 gives '-' sign for n%4!=0 width
  int f_ind = tx + ty * n + tz * n * n;
  int g_ind = (tx + n / 2 + m) + (ty+1 + n / 2 + m) * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m); // ty + 1 adjust for tomopy

  f[f_ind].x = g[g_ind].x * phi;
  f[f_ind].y = g[g_ind].y * phi;
}

void __global__ takexy(float *x, float *y, float *theta, int n, int nproj)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;

  if (tx >= n || ty >= nproj)
    return;
  x[tx + ty * n] = (tx - n / 2) / (float)n * __cosf(theta[ty]);
  y[tx + ty * n] = -(tx - n / 2) / (float)n * __sinf(theta[ty]);
  if (x[tx + ty * n] >= 0.5f)
    x[tx + ty * n] = 0.5f - 1e-5;
  if (y[tx + ty * n] >= 0.5f)
    y[tx + ty * n] = 0.5f - 1e-5;
}

void __global__ wrap(real2 *f, int n, int nz, int m)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= 2 * n + 2 * m || ty >= 2 * n + 2 * m || tz >= nz)
    return;
  if (tx < m || tx >= 2 * n + m || ty < m || ty >= 2 * n + m)
  {
    int tx0 = (tx - m + 2 * n) % (2 * n);
    int ty0 = (ty - m + 2 * n) % (2 * n);
    int id1 = tx + ty * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
    int id2 = tx0 + m + (ty0 + m) * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
#ifdef HALF
    atomicAdd(&f[id2], f[id1]);
#else
    atomicAdd(&f[id2].x, f[id1].x);
    atomicAdd(&f[id2].y, f[id1].y);
#endif
  }
}

// Binned scatter: `gather` with the loop inverted.  One block per (tile of the
// padded grid, z), accumulating in shared memory, so global memory is touched
// once per cell instead of once per atomic.  Ported from ~/tomo_usfft.

// Stencil-to-tile mapping for the two index passes below.  2m+1 <= b, so a
// stencil spans at most 2 tiles per axis; `t` gets their ids, count returned.
inline __device__ int sample_tiles(int *t, float x0, float y0, int m, int n,
                                   int b, int ntx)
{
  int px = n + (int)floorf(2 * n * x0);   // first stencil column, = n+m+base0
  int py = n + (int)floorf(2 * n * y0);
  int ix0 = px / b, ix1 = (px + 2 * m) / b;
  int iy0 = py / b, iy1 = (py + 2 * m) / b;
  int k = 0;
  t[k++] = iy0 * ntx + ix0;
  if (ix1 != ix0)                 t[k++] = iy0 * ntx + ix1;
  if (iy1 != iy0)                 t[k++] = iy1 * ntx + ix0;
  if (ix1 != ix0 && iy1 != iy0)   t[k++] = iy1 * ntx + ix1;
  return k;
}

void __global__ count_tiles(int *counts, const float *x, const float *y, int m,
                            int n, int nproj, int b, int ntx)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  if (tx >= n || ty >= nproj)
    return;
  int t[4];
  int k = sample_tiles(t, x[tx + ty * n], y[tx + ty * n], m, n, b, ntx);
  for (int i = 0; i < k; i++)
    atomicAdd(&counts[t[i]], 1);
}

// Counting sort, second pass: each (sample, tile) pair claims a slot in its
// tile's run.  Order within a run is arbitrary; it only has to keep a warp's
// staged reads in the same cache lines, which the atomics already do.
void __global__ fill_tiles(int *bin_samples, int *cursor, const int *offsets,
                           const float *x, const float *y, int m, int n,
                           int nproj, int b, int ntx)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  if (tx >= n || ty >= nproj)
    return;
  int t[4];
  int k = sample_tiles(t, x[tx + ty * n], y[tx + ty * n], m, n, b, ntx);
  for (int i = 0; i < k; i++)
    bin_samples[offsets[t[i]] + atomicAdd(&cursor[t[i]], 1)] = tx + ty * n;
}

#define SCATTER_CH 256   // samples staged per pass == blockDim.x

void __global__ scatter_binned(real2 *f, const real2 *__restrict__ g,
                               const float *__restrict__ x,
                               const float *__restrict__ y,
                               const int *__restrict__ bin_samples,
                               const int *__restrict__ sub_tile,
                               const int *__restrict__ sub_beg,
                               const int *__restrict__ sub_end,
                               const int *__restrict__ sub_scratch,
                               float2 *scratch,
                               int m, float mu, int n, int nproj, int nz,
                               int b, int ntx, int s)
{
  // Two planes with row stride b+1, not interleaved real2: interleaved, every
  // .x address is even and a warp reaches only 16 of the 32 banks.
  extern __shared__ float sh[];      // 2 * b * (b+1) floats
  __shared__ int ss[SCATTER_CH];     // staged sample indices
  __shared__ float2 sg[SCATTER_CH];  // staged sinogram values
  const int sub = blockIdx.x;
  const int tz = blockIdx.y;
  const int nt = b * (b + 1);
  float *const shx = sh;
  float *const shy = sh + nt;

  for (int i = threadIdx.x; i < 2 * nt; i += blockDim.x)
    sh[i] = 0.0f;

  const int tile = sub_tile[sub];
  const int tx0 = (tile % ntx) * b;
  const int ty0 = (tile / ntx) * b;

  const float coeff0 = PI / (mu * (4 * n * n));
  const float coeff1 = -PI * PI / mu;
  const float scale = 4.0f / n;
  const int len = 2 * m + 1;
  const int beg = sub_beg[sub], end = sub_end[sub];

  // Staged samples are consecutive detector bins, 2 grid cells apart, so
  // consecutive lanes would overlap.  Stage with lane j (coalesced), process
  // entry perm(j): 8 entries = 16 cells apart, wider than the stencil.
  const int jperm = (threadIdx.x & 31) * (SCATTER_CH / 32) + (threadIdx.x >> 5);

  for (int base = beg; base < end; base += SCATTER_CH)
  {
    const int cnt = min(SCATTER_CH, end - base);
    __syncthreads();
    if (threadIdx.x < cnt)
    {
      int s0 = bin_samples[base + threadIdx.x];
      ss[threadIdx.x] = s0;
      real2 v = g[s0 + tz * n * nproj];
      sg[threadIdx.x] = make_float2((float)v.x * scale, (float)v.y * scale);
    }
    __syncthreads();
    if (jperm < cnt)
    {
      const int smp = ss[jperm];
      const float x0 = x[smp], y0 = y[smp];
      const int base0 = (int)floorf(2 * n * x0) - m;
      const int base1 = (int)floorf(2 * n * y0) - m;
      const int px = n + m + base0, py = n + m + base1;

      // The stencil's part inside this tile is contiguous: with lx = i0 - offx
      // it is [0, len) clamped to [offx, offx + b).
      const int offx = tx0 - px;
      const int ia = max(0, offx), ibx = min(len, offx + b);
      const int offy = ty0 - py;
      const int ja = max(0, offy), jb = min(len, offy + b);
      if (ibx > ia && jb > ja)
      {
        float wxs[16];   // 2m+1 <= 16 at every size in use
        for (int i0 = ia; i0 < ibx; i0++)
        {
          float w0 = (base0 + i0) / (float)(2 * n) - x0;
          wxs[i0 - ia] = __expf(coeff1 * w0 * w0);
        }
        const float2 g0 = sg[jperm];
        for (int i1 = ja; i1 < jb; i1++)
        {
          float w1 = (base1 + i1) / (float)(2 * n) - y0;
          float ey = coeff0 * __expf(coeff1 * w1 * w1);
          int row = (i1 - offy) * (b + 1) - offx;
          for (int i0 = ia; i0 < ibx; i0++)
          {
            float w = wxs[i0 - ia] * ey;
            int o = row + i0;
            atomicAdd(&shx[o], w * g0.x);
            atomicAdd(&shy[o], w * g0.y);
          }
        }
      }
    }
  }
  __syncthreads();

  // An unsplit tile is owned by this block, so it is stored, not accumulated:
  // one rounding for the whole sum.  f was zeroed, covering untouched tiles.
  const int slot = sub_scratch[sub];   // < 0 unless this tile is split

#ifdef HALF
  // Split tiles must not meet in f -- every partial sum would round to 11 bits,
  // and the dense tiles that split are the central, largest ones.  They meet in
  // a float32 scratch tile and finalize_tiles rounds once.
  if (slot >= 0)
  {
    float2 *stile = scratch + ((size_t)slot * nz + tz) * b * b;
    for (int c0 = threadIdx.x; c0 < b * b; c0 += blockDim.x)
    {
      int lx = c0 % b, ly = c0 / b;
      float vx = shx[ly * (b + 1) + lx], vy = shy[ly * (b + 1) + lx];
      if (vx != 0.0f) atomicAdd(&stile[c0].x, vx);
      if (vy != 0.0f) atomicAdd(&stile[c0].y, vy);
    }
    return;
  }
#endif

  real2 *ftile = f + (size_t)tz * s * s;
  for (int c0 = threadIdx.x; c0 < b * b; c0 += blockDim.x)
  {
    int lx = c0 % b, ly = c0 / b;
    int gx = tx0 + lx, gy = ty0 + ly;
    if (gx >= s || gy >= s)   // the last tile row/column overhangs the grid
      continue;
    float vx = shx[ly * (b + 1) + lx], vy = shy[ly * (b + 1) + lx];
    size_t o = (size_t)gy * s + gx;
#ifdef HALF
    ftile[o] = __floats2half2_rn(vx, vy);
#else
    // float32 loses nothing accumulating in f, so no scratch is allocated
    if (slot >= 0)
    {
      if (vx != 0.0f) atomicAdd(&ftile[o].x, vx);
      if (vy != 0.0f) atomicAdd(&ftile[o].y, vy);
    }
    else
      ftile[o] = make_float2(vx, vy);
#endif
  }
}

// The split tiles' float32 scratch, rounded into the spectrum.  One block per
// (split tile, z), which owns the tile, so this is a store and not an atomic.
void __global__ finalize_tiles(real2 *f, const float2 *__restrict__ scratch,
                               const int *__restrict__ fin_tile, int nz, int b,
                               int ntx, int s)
{
  const int slot = blockIdx.x;
  const int tz = blockIdx.y;
  const int tile = fin_tile[slot];
  const int tx0 = (tile % ntx) * b;
  const int ty0 = (tile / ntx) * b;
  const float2 *stile = scratch + ((size_t)slot * nz + tz) * b * b;
  real2 *ftile = f + (size_t)tz * s * s;
  for (int c0 = threadIdx.x; c0 < b * b; c0 += blockDim.x)
  {
    int gx = tx0 + c0 % b, gy = ty0 + c0 / b;
    if (gx >= s || gy >= s)
      continue;
    float2 v = stile[c0];
    size_t o = (size_t)gy * s + gx;
#ifdef HALF
    ftile[o] = __floats2half2_rn(v.x, v.y);
#else
    ftile[o] = v;
#endif
  }
}

void __global__ ifftshiftc(real2 *f, int n, int nproj, int nz)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n || ty >= nproj || tz >= nz)
    return;
  real g = static_cast<real>(1 - 2 * ((tx + 1) % 2));
  int f_ind = tx + ty * n + tz * n * nproj;
  f[f_ind].x *= g;
  f[f_ind].y *= g;
}


void __global__ mulc(real2 *f, float c, int n, int nproj, int nz)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n || ty >= nproj || tz >= nz)
    return;
  int f_ind = tx + ty * n + tz * n * nproj;
  f[f_ind].x = static_cast<real>((float)f[f_ind].x * c);
  f[f_ind].y = static_cast<real>((float)f[f_ind].y * c);
}



void __global__ fftshiftc(real2 *f, int n, int nz)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n || ty >= n || tz >= nz)
    return;
  real g = static_cast<real>((1 - 2 * ((tx + 1) % 2)) * (1 - 2 * ((ty + 1) % 2)));
  f[tx + ty * n + tz * n * n].x *= g;
  f[tx + ty * n + tz * n * n].y *= g;
}

void __global__ circ(real2 *f, float r, int n, int nz)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n || ty >= n || tz >= nz)
    return;
  int id0 = tx + ty * n + tz * n * n;
  float x = (tx - n / 2) / float(n);
  float y = (ty - n / 2) / float(n);
  real lam = static_cast<real>((4 * x * x + 4 * y * y) < 1 - r);
  f[id0].x *= lam;
  f[id0].y *= lam;
}
