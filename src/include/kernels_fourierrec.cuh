#define PI 3.1415926535

// Divide by phi
void __global__ divphi(float2 *g, float2 *f, float mu, int N, int Nz, int m, dir direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= N || ty >= N || tz >= Nz)
    return;
  float phi = __expf(
    -mu * (tx - N / 2) * (tx - N / 2)
    -mu * (ty - N / 2) * (ty - N / 2)
  );
  int f_ind = (
    + tx
    + ty * N
    + tz * N * N
  );
  int g_ind = (
    + (tx + N / 2 + m)
    + (N - 1 - ty + N / 2 + m) * (2 * N + 2 * m)// N - 1 - ty  ---consistency with tomopy
    + tz * (2 * N + 2 * m) * (2 * N + 2 * m)
  );  
  f[f_ind].x = g[g_ind].x / phi / (4 * N * N);
  f[f_ind].y = g[g_ind].y / phi / (4 * N * N);  
}


void __global__ takexy(float *x, float *y, float *theta, int N, int Ntheta) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;

  if (tx >= N || ty >= Ntheta)
    return;
  x[tx + ty * N] = (tx - N / 2) / (float)N * __cosf(theta[ty]);
  y[tx + ty * N] = -(tx - N / 2) / (float)N * __sinf(theta[ty]);
  if (x[tx + ty * N] >= 0.5f)
    x[tx + ty * N] = 0.5f - 1e-5;
  if (y[tx + ty * N] >= 0.5f)
    y[tx + ty * N] = 0.5f - 1e-5;
}

void __global__ wrap(float2 *f, int N, int Nz, int M, dir direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= 2 * N + 2 * M || ty >= 2 * N + 2 * M || tz >= Nz)
    return;
  if (tx < M || tx >= 2 * N + M || ty < M || ty >= 2 * N + M) {
    int tx0 = (tx - M + 2 * N) % (2 * N);
    int ty0 = (ty - M + 2 * N) % (2 * N);
    int id1 = (
      + tx
      + ty * (2 * N + 2 * M)
      + tz * (2 * N + 2 * M) * (2 * N + 2 * M)
    );
    int id2 = (
      + tx0
      + M
      + (ty0 + M) * (2 * N + 2 * M)
      + tz * (2 * N + 2 * M) * (2 * N + 2 * M)
    );
    if (direction == TOMO_FWD) {
      f[id1].x = f[id2].x;
      f[id1].y = f[id2].y;
    } else {
      atomicAdd(&f[id2].x, f[id1].x);
      atomicAdd(&f[id2].y, f[id1].y);
    }
  }
}

void __global__ gather(float2 *g, float2 *f, float *x, float *y, int M,
                       float mu, int N, int Ntheta, int Nz, dir direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= N || ty >= Ntheta || tz >= Nz)
    return;

  float2 g0;
  float x0 = x[tx + ty * N];
  float y0 = y[tx + ty * N];
  int g_ind = (
    + tx
    + ty * N
    + tz * N * Ntheta
  );
  if (direction == TOMO_FWD) {
    g0.x = 0.0f;
    g0.y = 0.0f;
  } else {
    g0.x = g[g_ind].x / N;
    g0.y = g[g_ind].y / N;
  }
  for (int i1 = 0; i1 < 2 * M + 1; i1++) {
    int ell1 = floorf(2 * N * y0) - M + i1;
    for (int i0 = 0; i0 < 2 * M + 1; i0++) {
      int ell0 = floorf(2 * N * x0) - M + i0;
      float w0 = ell0 / (float)(2 * N) - x0;
      float w1 = ell1 / (float)(2 * N) - y0;
      float w = (
        PI / (sqrtf(mu * mu))
        * __expf(-PI * PI / mu * (w0 * w0) - PI * PI / mu * (w1 * w1))
      );
      int f_ind = (
        + N + M + ell0
        + (2 * N + 2 * M) * (N + M + ell1)
        + tz * (2 * N + 2 * M) * (2 * N + 2 * M)
      );
      if (direction == TOMO_FWD) {
        g0.x += w * f[f_ind].x;
        g0.y += w * f[f_ind].y;
      } else {
        float *fx = &(f[f_ind].x);
        float *fy = &(f[f_ind].y);
        atomicAdd(fx, w * g0.x);
        atomicAdd(fy, w * g0.y);
      }
    }
  }
  if (direction == TOMO_FWD){
    g[g_ind].x = g0.x / N;
    g[g_ind].y = g0.y / N;
  }
}


void __global__ ifftshiftc(float2 *f, int N, int Ntheta, int Nz) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= N || ty >= Ntheta || tz >= Nz)
    return;
  int g = (1 - 2 * ((tx + 1) % 2));
  int f_ind = tx + ty * N + tz * N * Ntheta;
  f[f_ind].x *= g;
  f[f_ind].y *= g;
}

void __global__ fftshiftc(float2 *f, int N, int Nz) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= N || ty >= N || tz >= Nz)
    return;
  int g = (1 - 2 * ((tx + 1) % 2)) * (1 - 2 * ((ty + 1) % 2));
  f[tx + ty * N + tz * N * N].x *= g;
  f[tx + ty * N + tz * N * N].y *= g;
}

