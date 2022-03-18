#define PI 3.1415926535

// Divide by phi
void __global__ divphi(float2 *g, float2 *f, float mu, int n, int nz, int ntheta, int m) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n || ty >= n || tz >= nz)
    return;
  float phi = __expf(
    -mu * (tx - n / 2) * (tx - n / 2)
    -mu * (ty - n / 2) * (ty - n / 2)
  );
  int f_ind = (
    + tx
    + ty * n
    + tz * n * n
  );
  int g_ind = (
    + (tx + n / 2 + m)
    + (ty + n / 2 + m) * (2 * n + 2 * m)
    + tz * (2 * n + 2 * m) * (2 * n + 2 * m)
  );  
  f[f_ind].x = g[g_ind].x / phi / (4 * n * n * ntheta);//added /ntheta
  f[f_ind].y = g[g_ind].y / phi / (4 * n * n * ntheta);  
}


void __global__ takexy(float *x, float *y, float *theta, int n, int ntheta) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;

  if (tx >= n || ty >= ntheta)
    return;
  x[tx + ty * n] = (tx - n / 2) / (float)n * __cosf(theta[ty]);
  y[tx + ty * n] = -(tx - n / 2) / (float)n * __sinf(theta[ty]);
  if (x[tx + ty * n] >= 0.5f)
    x[tx + ty * n] = 0.5f - 1e-5;
  if (y[tx + ty * n] >= 0.5f)
    y[tx + ty * n] = 0.5f - 1e-5;
}

void __global__ wrap(float2 *f, int n, int nz, int m) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= 2 * n + 2 * m || ty >= 2 * n + 2 * m || tz >= nz)
    return;
  if (tx < m || tx >= 2 * n + m || ty < m || ty >= 2 * n + m) {
    int tx0 = (tx - m + 2 * n) % (2 * n);
    int ty0 = (ty - m + 2 * n) % (2 * n);
    int id1 = (
      + tx
      + ty * (2 * n + 2 * m)
      + tz * (2 * n + 2 * m) * (2 * n + 2 * m)
    );
    int id2 = (
      + tx0
      + m
      + (ty0 + m) * (2 * n + 2 * m)
      + tz * (2 * n + 2 * m) * (2 * n + 2 * m)
    );
    atomicAdd(&f[id2].x, f[id1].x);
    atomicAdd(&f[id2].y, f[id1].y);  
  }
}

void __global__ gather(float2 *g, float2 *f, float *x, float *y, int m,
                       float mu, int n, int ntheta, int nz) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= n || ty >= ntheta || tz >= nz)
    return;

  float2 g0;
  float x0 = x[tx + ty * n];
  float y0 = y[tx + ty * n];
  int g_ind = (
    + tx
    + ty * n
    + tz * n * ntheta
  );
  g0.x = g[g_ind].x / n;
  g0.y = g[g_ind].y / n;
  for (int i1 = 0; i1 < 2 * m + 1; i1++) {
    int ell1 = floorf(2 * n * y0) - m + i1;
    for (int i0 = 0; i0 < 2 * m + 1; i0++) {
      int ell0 = floorf(2 * n * x0) - m + i0;
      float w0 = ell0 / (float)(2 * n) - x0;
      float w1 = ell1 / (float)(2 * n) - y0;
      float w = (
        PI / (sqrtf(mu * mu))
        * __expf(-PI * PI / mu * (w0 * w0) - PI * PI / mu * (w1 * w1))
      );
      int f_ind = (
        + n + m + ell0
        + (2 * n + 2 * m) * (n + m + ell1)
        + tz * (2 * n + 2 * m) * (2 * n + 2 * m)
      );
      float *fx = &(f[f_ind].x);
      float *fy = &(f[f_ind].y);
      atomicAdd(fx, w * g0.x);
      atomicAdd(fy, w * g0.y);      
    }
  }  
}


void __global__ ifftshiftc(float2 *f, int n, int ntheta, int nz) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n || ty >= ntheta || tz >= nz)
    return;
  int g = (1 - 2 * ((tx + 1) % 2));
  int f_ind = tx + ty * n + tz * n * ntheta;
  f[f_ind].x *= g;
  f[f_ind].y *= g;
}

void __global__ fftshiftc(float2 *f, int n, int nz) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n || ty >= n || tz >= nz)
    return;
  int g = (1 - 2 * ((tx + 1) % 2)) * (1 - 2 * ((ty + 1) % 2));
  f[tx + ty * n + tz * n * n].x *= g;
  f[tx + ty * n + tz * n * n].y *= g;
}


void __global__ circ(float2 *f, float r, int n, int nz) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n || ty >= n || tz >= nz)
    return;
  int id0 = tx + ty * n + tz * n * n;
  float x = (tx - n / 2) / float(n);
  float y = (ty - n / 2) / float(n);
  int lam = (4 * x * x + 4 * y * y) < 1 - r;
  f[id0].x *= lam;
  f[id0].y *= lam;
}