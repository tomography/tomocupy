#define PI 3.1415926535897932384626433


// Divide by phi
void __global__ take_x(float *x, float *y, float* theta, float phi, int k, int deth0, int detw, int deth, int ntheta) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= detw || ty >= deth || tz >= ntheta)
    return;
  float ku = (tx-detw/2.0f)/detw;
  float kv = (ty+k*deth-deth0/2.0f)/deth0;
  int ind = tx + ty*detw + tz*detw*deth;
  x[ind] = ku*cosf(theta[tz])+kv*sinf(theta[tz])*cosf(phi);
  y[ind] = ku*sinf(theta[tz])-kv*cosf(theta[tz])*cosf(phi);
  x[ind] = min(max(x[ind],-0.5f+1e-5),0.5f-1e-5);
  y[ind] = min(max(y[ind],-0.5f+1e-5),0.5f-1e-5);
}
// Divide by phi
void __global__ divker2d(float2 *g, float2 *f, int n0, int n1, int n2, int m0,
                         int m1, float mu0, float mu1, bool direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= n2)
    return;
  float ker = __expf(-mu0 * (tx - n0 / 2) * (tx - n0 / 2) -
                     mu1 * (ty - n1 / 2) * (ty - n1 / 2));
  int f_ind = tx + tz * n0 + ty * n0 * n2;
  int g_ind = tx + n0 / 2 + m0 + (n1-ty-1 + n1 / 2 + m1) * (2 * n0 + 2 * m0) +    //n1-ty-1 instead of ty for consistency with other methods
              tz * (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1);
  
  if (direction == 0){
    g[g_ind].x = f[f_ind].x / ker / (n0 * n1);
    g[g_ind].y = f[f_ind].y / ker / (n0 * n1);
  } else {
    f[f_ind].x = g[g_ind].x / ker / (n0 * n1);
    f[f_ind].y = g[g_ind].y / ker / (n0 * n1);
  }
}

void __global__ fftshiftc2d(float2 *f, int n0, int n1, int n2) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= n2)
    return;
  int g = (1 - 2 * ((tx + 1) % 2)) * (1 - 2 * ((ty + 1) % 2));
  f[tx + ty * n0 + tz * n0 * n1].x *= g;
  f[tx + ty * n0 + tz * n0 * n1].y *= g;
}

void __global__ wrap2d(float2 *f, int n0, int n1, int n2, int m0, int m1, bool direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= 2 * n0 + 2 * m0 || ty >= 2 * n1 + 2 * m1 || tz >= n2)
    return;
  if (tx < m0 || tx >= 2 * n0 + m0 || ty < m1 || ty >= 2 * n1 + m1) {
    int tx0 = (tx - m0 + 2 * n0) % (2 * n0);
    int ty0 = (ty - m1 + 2 * n1) % (2 * n1);
    int id1 = (+tx + ty * (2 * n0 + 2 * m0) +
               tz * (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1));
    int id2 = (+tx0 + m0 + (ty0 + m1) * (2 * n0 + 2 * m0) +
               tz * (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1));
    if (direction == 0) {
      f[id1].x = f[id2].x;
      f[id1].y = f[id2].y;
    } else {
      atomicAdd(&f[id2].x, f[id1].x);
      atomicAdd(&f[id2].y, f[id1].y);
    }
  }
}
void __global__ gather2d(float2 *g, float2 *f, float *x, float *y, int m0,
                         int m1, float mu0, float mu1, int n0, int n1, int n2,
                         int detw, int deth, int ntheta, bool direction) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= detw || ty >= deth || tz >= ntheta)
    return;

  int txs = tx;
  int tys = ty; 
  int tzs = tz; 
  int conj = 1;
  if (tx>detw/2) 
  {    
    txs = txs - 2*(tx-detw/2);   
    tys = deth - ty-1;
    tzs = tz + ntheta;
    conj = -1;
  }
  int g_ind = txs + tys*(detw/2+1) + tzs*(detw/2+1)*deth;
  int xy_ind = tx + ty * detw + tz* detw * deth;

  
  float x0 = x[xy_ind];
  float y0 = y[xy_ind];

  float2 g0;
  if (direction == 0) {
    g0.x = 0.0f;
    g0.y = 0.0f;
  } else {
    g0.x = g[g_ind].x;
    g0.y = conj*g[g_ind].y;
  }
  for (int i1 = 0; i1 < 2 * m1 + 1; i1++) {
    int ell1 = floorf(2 * n1 * y0) - m1 + i1;
    for (int i0 = 0; i0 < 2 * m0 + 1; i0++) {
      int ell0 = floorf(2 * n0 * x0) - m0 + i0;
      float w0 = ell0 / (float)(2 * n0) - x0;
      float w1 = ell1 / (float)(2 * n1) - y0;
      float w = PI / sqrtf(mu0 * mu1 * ntheta) *
                __expf(-PI * PI / mu0 * (w0 * w0) - PI * PI / mu1 * (w1 * w1));
      int f_ind = n0 + m0 + ell0 + (2 * n0 + 2 * m0) * (n1 + m1 + ell1) +
                  (2 * n0 + 2 * m0) * (2 * n1 + 2 * m1) * ty;
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
  }
  if (direction == 0){
    g[g_ind].x = g0.x;
    g[g_ind].y = g0.y;
  }
}
