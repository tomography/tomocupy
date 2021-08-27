#include<cfunc.cuh>
#define Pole static_cast<real>(-0.267949192431123f)

texture<float, cudaTextureType2DLayered, cudaReadModeElementType> texg; //type 0
texture<float, cudaTextureType2DLayered, cudaReadModeElementType> texfl; //type 1

//CODE adapted from https://github.com/nikitinvv/lprec/blob/master/src/include/main_kernels.cuh

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

__device__ real linearTex2D(texture<float, cudaTextureType2DLayered, cudaReadModeElementType> tex, float x, float y, float z, int n0,int n1)
{
	return TEX2D_L(tex, (x+0.5f)/float(n0), (y+0.5f)/float(n1), z);
}

//cubic interpolation via two linear interpolations for several slices, texture is not normalized
__device__ real cubicTex2D(texture<float, cudaTextureType2DLayered, cudaReadModeElementType> tex, float x, float y, float z, int n0,int n1)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	real w[4] = {};
	const float indexx = floor(x);
	const real fractionx = real(x - indexx); //big-big number
	bspline_weights(fractionx, w);
	const real g0x = (w[0] + w[1]);
	const real g1x = (w[2] + w[3]);
	const float h0x = (float(w[1] / g0x - static_cast<real>(0.5f))+float(indexx))/float(n0);  //big+small number
	const float h1x = (float(w[3] / g1x + static_cast<real>(1.5f))+float(indexx))/float(n0);  
	
	
	const float indexy = floor(y);
	const real fractiony = real(y - indexy);
	bspline_weights(fractiony, w);
	const real g0y = w[0] + w[1];
	const real g1y = w[2] + w[3];
	const float h0y = (float(w[1] / g0y - static_cast<real>(0.5f))+float(indexy))/float(n1);  
	const float h1y = (float(w[3] / g1y + static_cast<real>(1.5f))+float(indexy))/float(n1);  
	

	real tex00 = TEX2D_L(tex, h0x, h0y, z);
	real tex10 = TEX2D_L(tex, h1x, h0y, z);
	real tex01 = TEX2D_L(tex, h0x, h1y, z);
	real tex11 = TEX2D_L(tex, h1x, h1y, z);

	tex00 = g0y * tex00 + g1y * tex01;
	tex10 = g0y * tex10 + g1y * tex11;
	
	return g0x * tex00 + g1x * tex10;
}

__global__ void interp(int interp_id, real *f, float* x, float* y, int w, int np,int n1,int n2, int nz, int* cids, int step2d)
{
	uint tx = blockIdx.x*blockDim.x + threadIdx.x;
	uint ty = blockIdx.y*blockDim.y + threadIdx.y;
	uint tz = blockIdx.z*blockDim.z + threadIdx.z;
	uint tid = ty*w+tx;
	if(tid>=np||tz>=nz) return;
	
	switch(interp_id)//no overhead, all threads have the same way
	{ 		
		case 0: f[tz*step2d+cids[tid]] += linearTex2D(texg, x[tid], y[tid], tz,n1,n2);break;   
		case 1: f[tz*step2d+cids[tid]] += linearTex2D(texfl, x[tid], y[tid], tz,n1,n2);break;   
		case 2: f[tz*step2d+cids[tid]] += cubicTex2D(texg, x[tid], y[tid], tz,n1,n2);break;   
		case 3: f[tz*step2d+cids[tid]] += cubicTex2D(texfl, x[tid], y[tid], tz,n1,n2);break;  
	}	
}

//casual cofficients for prefilter
__device__ real InitialCausalCoefficient(real* c, uint DataLength,int step)
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
__device__ real InitialAntiCausalCoefficient(real* c,uint DataLength,int step)
{
	// this initialization corresponds to clamping boundaries
	return((Pole / (Pole - static_cast<real>(1.0f))) * *c);
}

//compute coefficients from samples c
__device__ void ConvertToInterpolationCoefficients(real* coeffs,uint DataLength,int step)
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
