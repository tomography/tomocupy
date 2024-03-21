#define PI 3.1415926535897932384626433


void __global__ take_x(float *x, float phi, int deth) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tx >= deth)
    return;
  x[tx] = (tx - deth/2.0f)/deth*sinf(phi);
}

// Divide by phi
void __global__ divker1d(float2 *g, float *f, int n0, int n1, int n2, int m2, float mu2, bool direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= n2)
    return;
  float ker = __expf(-mu2 * (tz - n2 / 2) * (tz - n2 / 2));
  int f_ind = tx + tz * n0 + ty * n0 * n2;
  int g_ind = tx + ty * n0 + (tz + n2 / 2 + m2) * n0 * n1;
  

  // if (n2%2!=0) ker=-ker;// handle sizes not multiples of 4

  if (direction == 0){
    g[g_ind].x = f[f_ind] / ker / (2 * n2);
  } else {
    f[f_ind] = g[g_ind].x / ker / (2 * n2);
  }
}

void __global__ fftshiftc1d(float2 *f, int n0, int n1, int n2) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= n2)
    return;
  int g = (1 - 2 * ((tz + 1) % 2));  
  int f_ind = tx + ty * n0 + tz * n0 * n1;
  f[f_ind].x *= g;
  f[f_ind].y *= g;
}

void __global__ wrap1d(float2 *f, int n0, int n1, int n2, int m2, bool direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= 2 * n2 + 2 * m2)
    return;
  if (tz < m2 || tz >= 2 * n2 + m2) {
    int tz0 = (tz - m2 + 2 * n2) % (2 * n2);
    int id1 = (+tx + ty * n0 + tz * n0 * n1);
    int id2 = (+tx + ty * n0 + (tz0 + m2) * n0 * n1);
    if (direction == 0) {
      f[id1].x = f[id2].x;
      f[id1].y = f[id2].y;
    } else {
      atomicAdd(&f[id2].x, f[id1].x);
      atomicAdd(&f[id2].y, f[id1].y);
    }
  }
}
void __global__ gather1d(float2 *g, float2 *f, float *z, int m2, float mu2,
                         int n0, int n1, int n2, int deth, bool direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= n0 || ty >= n1 || tz >= deth)
    return;

  float2 g0;
  float z0 = z[tz];

  int tzs = tz;
  int conj = 1;
  if (tz>deth/2) 
  {
    tzs = tzs - 2*(tz-deth/2);
    conj=-1;
  }

  int g_ind = tx + tzs * n0 + ty * n0 * (deth/2+1);

  if (direction == 0) {
    g0.x = 0.0f;
    g0.y = 0.0f;
  } else {
    g0.x = g[g_ind].x;
    g0.y = conj*g[g_ind].y;
  }

  for (int i2 = 0; i2 < 2 * m2 + 1; i2++) {
    int ell2 = floorf(2 * n2 * z0) - m2 + i2;
    float w2 = ell2 / (float)(2 * n2) - z0;
    float w = sqrtf(PI / (mu2*n0)) * __expf(-PI * PI / mu2 * (w2 * w2));
    int f_ind = tx + ty * n0 + (n2 + m2 + ell2) * n0 * n1;

    if (direction == 0) {
      g0.x += w * f[f_ind].x;
      g0.y += w * f[f_ind].y;
    } else {
      float *fx = &(f[f_ind].x);
      float *fy = &(f[f_ind].y);
      atomicAdd(fx, w * g0.x);
      atomicAdd(fy, w * g0.y);
    }
  }

  if (direction == 0){
    g[g_ind].x = g0.x;
    g[g_ind].y = g0.y;
  }
}