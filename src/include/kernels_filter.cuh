#include "defs.cuh"
#define PI 3.1415926535f

void __global__ mulw(real2 *g, real2 *w, int n, int nproj, int nz)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n || ty >= nproj || tz >= nz)
    return;
  int g_ind = tx + ty * n + tz * n * nproj;
  int w_ind = tx + tz * n;
  real2 g0;
  g0.x = g[g_ind].x * w[w_ind].x - g[g_ind].y * w[w_ind].y;
  g0.y = g[g_ind].x * w[w_ind].y + g[g_ind].y * w[w_ind].x;
  g[g_ind].x = g0.x;
  g[g_ind].y = g0.y;
}

void __global__ mulrec(real *f, float c, int n, int nproj, int nz)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n || ty >= nproj || tz >= nz)
    return;
  int f_ind = tx + ty * n + tz * n * nproj;
  f[f_ind] = static_cast<real>((float)f[f_ind] * c);
}
