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

void __global__ gather(real2 *g, real2 *f, float *x, float *y, int m,
                       float mu, int n, int nproj, int nz)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= n || ty >= nproj || tz >= nz)
    return;

  real2 g0, g0t;
  real w, coeff0;
  float w0, w1, x0, y0, coeff1;
  int ell0, ell1, g_ind, f_ind;

  g_ind = tx + ty * n + tz * n * nproj;
  coeff0 = static_cast<real>(PI / (mu * (4 * n * n)));
  coeff1 = -PI * PI / mu;
  x0 = x[tx + ty * n];
  y0 = y[tx + ty * n];
  g0.x = g[g_ind].x;
  g0.y = g[g_ind].y;
  for (int i1 = 0; i1 < 2 * m + 1; i1++)
  {
    ell1 = floorf(2 * n * y0) - m + i1;
    for (int i0 = 0; i0 < 2 * m + 1; i0++)
    {
      ell0 = floorf(2 * n * x0) - m + i0;
      w0 = ell0 / (float)(2 * n) - x0;
      w1 = ell1 / (float)(2 * n) - y0;
      w = coeff0 * mexp(static_cast<real>(coeff1 * (w0 * w0 + w1 * w1))); //the inner part is in float32 precision since involves large and small values
      g0t.x = w * g0.x;
      g0t.y = w * g0.y;
      f_ind = n + m + ell0 + (2 * n + 2 * m) * (n + m + ell1) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
#ifdef HALF
      atomicAdd(&(f[f_ind]), g0t);
#else
      atomicAdd(&(f[f_ind].x), g0t.x);
      atomicAdd(&(f[f_ind].y), g0t.y);
#endif
    }
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
