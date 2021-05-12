__global__ void interplp(float *f, float* g, float* x, float* y, int w, int np,int n1,int n2, int ni, int* cids, int step2d)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	uint tz = blockIdx.z*blockDim.z + threadIdx.z;
	uint tid = ty*w+tx;
	if(tid>=np||tz>=ni) return;

	int xc = (int)x[tid];
    int yc = (int)y[tid];
    float xf = x[tid]-xc;    
    float yf = y[tid]-yc;
    f[tz*step2d+cids[tid]] = g[tz*n1*n2+yc*n1+xc]*(1-xf)*(1-yf)
                    +g[tz*n1*n2+yc*n1+(xc+1)%n1]*xf*(1-yf)
                    +g[tz*n1*n2+((yc+1)%n2)*n1+xc]*(1-xf)*yf
                    +g[tz*n1*n2+((yc+1)%n2)*n1+(xc+1)%n1]*xf*yf;                                  
}

__global__ void interpc(float *f, float* g, float* x, float* y, int w, int np,int n1,int n2, int ni, int* cids, int step2d)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	uint tz = blockIdx.z*blockDim.z + threadIdx.z;
	uint tid = ty*w+tx;
	if(tid>=np||tz>=ni) return;

	int xc = (int)x[tid];
    int yc = (int)y[tid];
    float xf = x[tid]-xc;    
    float yf = y[tid]-yc;
    f[tz*step2d+cids[tid]] += g[tz*n1*n2+yc*n1+xc]*(1-xf)*(1-yf)
                    +g[tz*n1*n2+yc*n1+(xc+1)%n1]*xf*(1-yf)
                    +g[tz*n1*n2+((yc+1)%n2)*n1+xc]*(1-xf)*yf
                    +g[tz*n1*n2+((yc+1)%n2)*n1+(xc+1)%n1]*xf*yf;  
}

__global__ void mul(float c, float2* y,float2* x,int n1,int n2, int ni)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	uint tz = blockIdx.z*blockDim.z + threadIdx.z;
	if (tx>=n1||ty>=n2||tz>=ni) return;
	float2 x0,y0;
	y0=y[tz*n1*n2+ty*n1+tx];x0=x[ty*n1+tx];
	y[tz*n1*n2+ty*n1+tx].x=c*(y0.x*x0.x-y0.y*x0.y);
	y[tz*n1*n2+ty*n1+tx].y=c*(y0.x*x0.y+y0.y*x0.x);
}