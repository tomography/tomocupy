#include <stdint.h>
#include <stdio.h>
#include <cuda_fp16.h>
typedef half mt;  // use an integer type

__global__ void kernel(cudaTextureObject_t tex, mt* d)
{
    int x = threadIdx.x;
    int y = threadIdx.y;
    d[y*4+x] = mt(1);
    mt val = __float2half_rn(tex2D<float>(tex, x-1, y+0.5));
    printf("%f, ", float(val));
}

int main(int argc, char **argv)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("texturePitchAlignment: %lu\n", prop.texturePitchAlignment);
    cudaTextureObject_t tex;
    const int num_rows = 4;
    const int num_cols = prop.texturePitchAlignment*1; // should be able to use a different multiplier here
    const int ts = num_cols*num_rows;
    const int ds = ts*sizeof(ts);
    mt* dataDev = 0;
    cudaMalloc((void**)&dataDev, ds);
    
    
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = dataDev;
    resDesc.res.pitch2D.width = num_cols;
    resDesc.res.pitch2D.height = num_rows;
    resDesc.res.pitch2D.desc = cudaCreateChannelDescHalf(); 
    resDesc.res.pitch2D.pitchInBytes = num_cols*sizeof(mt);
    resDesc.addressMode[0] = cudaAddressModeClamp;
	resDesc.addressMode[1] = cudaAddressModeClamp;
    struct cudaTextureDesc texDesc;
    
    memset(&texDesc, 0, sizeof(texDesc));
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    
    
    
    dim3 threads(4, 4);
    kernel<<<1, threads>>>(tex, dataDev);    
    cudaDeviceSynchronize();


    printf("\n");
    return 0;
}