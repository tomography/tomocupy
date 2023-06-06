#include "defs.cuh"

// Efficient cubic texture filtering (adapted from https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering) 

//compute weigts for the cubic B spline
__device__ void bspline_weights(real fraction, real* w)
{
	const real one_frac = static_cast<real>(1.0f) - fraction;
	const real squared = fraction * fraction;
	const real one_sqd = one_frac * one_frac;

	w[0] = static_cast<real>(1.0f/6.0f) * one_sqd * one_frac;
	w[1] = static_cast<real>(2.0f/3.0f) - static_cast<real>(0.5f) * squared * (static_cast<real>(2.0f)-fraction);
	w[2] = static_cast<real>(2.0f/3.0f) - static_cast<real>(0.5f) * one_sqd * (static_cast<real>(2.0f)-one_frac);
	w[3] = static_cast<real>(1.0f/6.0f) * squared * fraction;
}

//cubic interpolation via two linear interpolations for several slices, texture is not normalized
__device__ real cubicTex2D(cudaTextureObject_t tex, float x, float y, float z, int n0,int n1)
{		
	//compute weigths and texture coordinates in x direction
	const float indexx = floor(x);
	const real fractionx = real(x - indexx); //big - big -> compute in floats
	real wx[4] = {};
	bspline_weights(fractionx, wx);
	const real g0x = (wx[0] + wx[1]);
	const real g1x = (wx[2] + wx[3]);
	const float h0x = (float(wx[1] / g0x - static_cast<real>(0.5f)) + indexx)/float(n0);  //big + small -> compute in floats
	const float h1x = (float(wx[3] / g1x + static_cast<real>(1.5f)) + indexx)/float(n0);  
	
	//compute weigths and texture coordinates in y direction
	const float indexy = floor(y);
	const real fractiony = real(y - indexy);
	real wy[4] = {};
	bspline_weights(fractiony, wy);
	const real g0y = wy[0] + wy[1];
	const real g1y = wy[2] + wy[3];
	const float h0y = (float(wy[1] / g0y - static_cast<real>(0.5f)) + indexy)/float(n1);  
	const float h1y = (float(wy[3] / g1y + static_cast<real>(1.5f)) + indexy)/float(n1);  
	
	// read from texture
	real tex00 = TEX2D_L(tex, h0x, h0y, z);
	real tex10 = TEX2D_L(tex, h1x, h0y, z);
	real tex01 = TEX2D_L(tex, h0x, h1y, z);
	real tex11 = TEX2D_L(tex, h1x, h1y, z);

	tex00 = g0y * tex00 + g1y * tex01;
	tex10 = g0y * tex10 + g1y * tex11;
	
	return g0x * tex00 + g1x * tex10;
}

// interpolation to irregular grid
__global__ void interp(cudaTextureObject_t tex, real *f, float* x, float* y, int w, int np,int n1,int n2, int nz, int* cids, int step2d)
{
	unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int tz = blockIdx.z*blockDim.z + threadIdx.z;
	unsigned int tid = ty*w+tx;
	if(tid>=np||tz>=nz) return;
	f[tz*step2d+cids[tid]] += cubicTex2D(tex, x[tid], y[tid], tz,n1,n2);	
}

//casual cofficients for prefilter
__device__ real InitialCausalCoefficient(real* c, unsigned int DataLength,int step)
{
	const unsigned int Horizon = 12<DataLength?12:DataLength;

	// this initialization corresponds to clamping boundaries
	// accelerated loop
	real zn = Pole;
	real Sum = *c;
	for (unsigned int n = 0; n < Horizon; n++) {
		Sum += zn * *c;
		zn *= Pole;
		c = (real*)((unsigned char*)c + step);
	}
	return(Sum);
}

//anticasual coffeicients for prefilter
__device__ real InitialAntiCausalCoefficient(real* c,unsigned int DataLength,int step)
{
	// this initialization corresponds to clamping boundaries
	return((Pole / (Pole - static_cast<real>(1.0f))) * *c);
}

//compute coefficients from samples c
__device__ void ConvertToInterpolationCoefficients(real* coeffs,unsigned int DataLength,int step)
{
	// compute the overall gain
	const real Lambda = (static_cast<real>(1.0f) - Pole) * (static_cast<real>(1.0f) - static_cast<real>(1.0f) / Pole);

	// causal initialization
	real* c = coeffs;
	real previous_c;  //cache the previously calculated c rather than look it up again (faster!)
	*c = previous_c = Lambda * InitialCausalCoefficient(c, DataLength, step);
	// causal recursion
	for (int n = 1; n < DataLength; n++) {
		c = (real*)((unsigned char*)c + step);
		*c = previous_c = Lambda * *c + Pole * previous_c;
	}
	// anticausal initialization
	*c = previous_c = InitialAntiCausalCoefficient(c, DataLength, step);
	// anticausal recursion
	for (int n = DataLength - 2; 0 <= n; n--) {
		c = (real*)((unsigned char*)c - step);
		*c = previous_c = Pole * (previous_c - *c);
	}
}

//fast transpose on GPU
__global__ void transpose(real *odata, real *idata, int width, int height, int nz)
{
	__shared__ real block[BS3][BS2][BS1+1];

	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int zIndex = blockIdx.z*blockDim.z + threadIdx.z;
	if(zIndex>=nz) return;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = zIndex*width*height+yIndex * width + xIndex;
		block[threadIdx.z][threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * blockDim.y + threadIdx.x;// could be wrong Dimx and Dimy
	yIndex = blockIdx.x * blockDim.x + threadIdx.y;// could be wrong Dimx and Dimy
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = zIndex*width*height+yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.z][threadIdx.x][threadIdx.y];
	}
}

// compute coefficients from samples only for rows (gpu cache optimization) 
__global__ void SamplesToCoefficients2DY(real* image,unsigned int pitch,unsigned int width,unsigned int height, int nz)
{
	// process lines in x-direction
	unsigned int yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	if(yIndex>=nz) return;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x>=width) return;

	real* line = yIndex*width*height+image + x;  //direct access

	ConvertToInterpolationCoefficients(line, height, pitch);
}

// real2 multiplication
__global__ void mul(real2* y,real2* x,int n1,int n2, int nz)
{
	unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int tz = blockIdx.z*blockDim.z + threadIdx.z;
	if (tx>=n1||ty>=n2||tz>=nz) return;
    real2 x0,y0;
    y0=y[tz*n1*n2+ty*n1+tx];x0=x[ty*n1+tx];
	y[tz*n1*n2+ty*n1+tx].x=y0.x*x0.x-y0.y*x0.y;
	y[tz*n1*n2+ty*n1+tx].y=y0.x*x0.y+y0.y*x0.x;
}

void __global__ mulc(real *f, float c, int n1, int n2, int n3)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  if (tx >= n1 || ty >= n2 || tz >= n3)
    return;
  int f_ind = tx + ty * n1 + tz * n1 * n2;
  f[f_ind] = static_cast<real>((float)f[f_ind] * c);  
}