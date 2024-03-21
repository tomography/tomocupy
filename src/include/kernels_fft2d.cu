void __global__ rfftshiftc2d(float *f, int n0, int n1, int n2) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= n2)
    return;
  int g = (1 - 2 * ((tx + 1) % 2)) * (1 - 2 * ((ty + 1) % 2));
  f[tx + ty * n0 + tz * n0 * n1] *= g;  
}

void __global__ irfftshiftc2d(float2 *f, int n0, int n1, int n2) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= n2)
    return;
  int g = -(1 - 2 * ((tx + 1) % 2)) * (1 - 2 * ((ty + 1) % 2));
  f[tx + ty * n0 + tz * n0 * n1].x *= g;
  f[tx + ty * n0 + tz * n0 * n1].y *= g;
}

void __global__ mulc(float2 *f, int n0, int n1, int n2, float c) {
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n0 || ty >= n1 || tz >= n2)
    return;
  
  f[tx + ty * n0 + tz * n0 * n1].x *= c ;
  f[tx + ty * n0 + tz * n0 * n1].y *= c;
}
