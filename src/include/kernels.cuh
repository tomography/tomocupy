#include<cfunc.cuh>
#include"math_operators.cuh"

#define Pole static_cast<real>(-0.267949192431123f)

texture<real, cudaTextureType2DLayered, cudaReadModeElementType> texg; //type 0
texture<real, cudaTextureType2DLayered, cudaReadModeElementType> texfl; //type 1

//CODE adapted from https://github.com/nikitinvv/lprec/blob/master/src/include/main_kernels.cuh

//compute weigts for the cubic B spline
__host__ __device__ void bspline_weights(complex fraction, complex& w0, complex& w1, complex& w2, complex& w3)
{
	const complex one_frac = static_cast<real>(1.0f) - fraction;
	const complex squared = fraction * fraction;
	const complex one_sqd = one_frac * one_frac;

	w0 = static_cast<real>(1.0f/6.0f) * one_sqd * one_frac;
	w1 = static_cast<real>(2.0f/3.0f) - static_cast<real>(0.5f) * squared * (static_cast<real>(2.0f)-fraction);
	w2 = static_cast<real>(2.0f/3.0f) - static_cast<real>(0.5f) * one_sqd * (static_cast<real>(2.0f)-one_frac);
	w3 = static_cast<real>(1.0f/6.0f) * squared * fraction;
}

__device__ real linearTex2D(texture<real, cudaTextureType2DLayered, cudaReadModeElementType> tex, real x, real y, real z, int n0,int n1)
{
	complex t0;
	t0.x = x/(real)n0;
	t0.y = y/(real)n1;
	return tex2DLayered(tex, t0.x, t0.y, z);
}

//cubic interpolation via two linear interpolations for several slices, texture is not normalized
__device__ real cubicTex2D(texture<real, cudaTextureType2DLayered, cudaReadModeElementType> tex, real x, real y, real z, int n0,int n1)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	const complex coord_grid = make_complex(x - static_cast<real>(0.5f), y - static_cast<real>(0.5f));
	const complex index = floor(coord_grid);
	const complex fraction = coord_grid - index;
	complex w0, w1, w2, w3;
	bspline_weights(fraction, w0, w1, w2, w3);

	const complex g0 = w0 + w1;
	const complex g1 = w2 + w3;
	const complex h0 = (w1 / g0) - make_complex(static_cast<real>(0.5f)) + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
	const complex h1 = (w3 / g1) + make_complex(static_cast<real>(1.5f)) + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

	complex t0,t1;
	t0.x = h0.x/(real)n0;
	t1.x = h1.x/(real)n0;
	t0.y = h0.y/(real)n1;
	t1.y = h1.y/(real)n1;
	real tex00 = tex2DLayered(tex, t0.x, t0.y, z);
	real tex10 = tex2DLayered(tex, t1.x, t0.y, z);
	real tex01 = tex2DLayered(tex, t0.x, t1.y, z);
	real tex11 = tex2DLayered(tex, t1.x, t1.y, z);


	// weigh along the y-direction
	tex00 = g0.y * tex00 + g1.y * tex01;
	tex10 = g0.y * tex10 + g1.y * tex11;

	// weigh along the x-direction
	return (g0.x * tex00 + g1.x * tex10);
}

__global__ void interp(int interp_id, real *f, float* x, float* y, int w, int np,int n1,int n2, int nz, int* cids, int step2d)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	uint tz = blockIdx.z*blockDim.z + threadIdx.z;
	uint tid = ty*w+tx;
	if(tid>=np||tz>=nz) return;
	real u = real(x[tid]+static_cast<real>(0.5f));
	real v = real(y[tid]+static_cast<real>(0.5f));
	switch(interp_id)//no overhead, all threads have the same way
	{ 		
		case 0: f[tz*step2d+cids[tid]] += linearTex2D(texg, u, v, tz,n1,n2);break;   
		case 1: f[tz*step2d+cids[tid]] += linearTex2D(texfl, u, v, tz,n1,n2);break;   
		case 2: f[tz*step2d+cids[tid]] += cubicTex2D(texg, u, v, tz,n1,n2);break;   
		case 3: f[tz*step2d+cids[tid]] += cubicTex2D(texfl, u, v, tz,n1,n2);break;  
	}	
}

//casual cofficients for prefilter
__host__ __device__ real InitialCausalCoefficient(real* c, uint DataLength,int step)
{
	const uint Horizon = 12<DataLength?12:DataLength;

	// this initialization corresponds to clamping boundaries
	// accelerated loop
	real zn = Pole;
	real Sum = *c;
	for (uint n = 0; n < Horizon; n++) {
		Sum += zn * *c;
		zn *= Pole;
		c = (real*)((unsigned char*)c + step);
	}
	return(Sum);
}

//anticasual coffeicients for prefilter
__host__ __device__ real InitialAntiCausalCoefficient(real* c,uint DataLength,int step)
{
	// this initialization corresponds to clamping boundaries
	return((Pole / (Pole - static_cast<real>(1.0f))) * *c);
}

//compute coefficients from samples c
__host__ __device__ void ConvertToInterpolationCoefficients(real* coeffs,uint DataLength,int step)
{
	// compute the overall gain
	const real Lambda = (static_cast<real>(1.0f) - Pole) * (static_cast<real>(1.0f) - static_cast<real>(1.0f) / Pole);

	// causal initialization
	real* c = coeffs;
	real previous_c;  //cache the previously calculated c rather than look it up again (faster!)
	*c = previous_c = Lambda * InitialCausalCoefficient(c, DataLength, step);
	// causal recursion
	for (uint n = 1; n < DataLength; n++) {
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
	uint xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	uint yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	uint zIndex = blockIdx.z*blockDim.z + threadIdx.z;
	if(zIndex>=nz) return;
	if((xIndex < width) && (yIndex < height))
	{
		uint index_in = zIndex*width*height+yIndex * width + xIndex;
		block[threadIdx.z][threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * blockDim.y + threadIdx.x;// could be wrong Dimx and Dimy
	yIndex = blockIdx.x * blockDim.x + threadIdx.y;// could be wrong Dimx and Dimy
	if((xIndex < height) && (yIndex < width))
	{
		uint index_out = zIndex*width*height+yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.z][threadIdx.x][threadIdx.y];
	}
}

// compute coefficients from samples only for rows (gpu cache optimization) 
__global__ void SamplesToCoefficients2DY(real* image,uint pitch,uint width,uint height, int nz)
{
	// process lines in x-direction
	uint yIndex = blockIdx.y*blockDim.y + threadIdx.y;
	if(yIndex>=nz) return;

	const uint x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x>=width) return;

	real* line = yIndex*width*height+image + x;  //direct access

	ConvertToInterpolationCoefficients(line, height, pitch);
}

__global__ void mul(complex* y,complex* x,int n1,int n2, int nz)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	uint tz = blockIdx.z*blockDim.z + threadIdx.z;
	if (tx>=n1||ty>=n2||tz>=nz) return;
    complex x0,y0;
    y0=y[tz*n1*n2+ty*n1+tx];x0=x[ty*n1+tx];
	y[tz*n1*n2+ty*n1+tx].x=y0.x*x0.x-y0.y*x0.y;
	y[tz*n1*n2+ty*n1+tx].y=y0.x*x0.y+y0.y*x0.x;
}
