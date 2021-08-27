#include "cfunc.cuh"
#include "kernels.cuh"
#include <stdio.h>

cudaError_t copy3DDeviceToArray(cudaArray* dfa, real* df, cudaExtent ext, cudaStream_t stream)
{
	cudaMemcpy3DParms param = { 0 };
	param.srcPtr   = make_cudaPitchedPtr((void*)df, ext.width*sizeof(real), ext.width, ext.height);
	param.dstArray = dfa;
	param.kind = cudaMemcpyDeviceToDevice;
	param.extent = ext;
	return cudaMemcpy3DAsync(&param,stream);
}

cfunc::cfunc(int nproj, int nz, int n, int ntheta, int nrho):
nproj(nproj), nz(nz), n(n), ntheta(ntheta), nrho(nrho) {

    // Create FFT plans for Fourier Transform in log-polar coordinates
    long long ffts[] = {nrho,ntheta};
	long long idist = nrho*ntheta;long long odist = nrho*(ntheta/2+1);
    long long inembed[] = {nrho, ntheta};long long onembed[] = {nrho, ntheta/2+1};
    size_t workSize = 0;
    cufftCreate(&plan_forward);
    cufftXtMakePlanMany(plan_forward, 
        2, ffts, 
        inembed, 1, idist, CUDA_R, 
        onembed, 1, odist, CUDA_C, 
        nz, &workSize, CUDA_C);    
    cufftCreate(&plan_inverse);
    cufftXtMakePlanMany(plan_inverse, 
        2, ffts, 
        onembed, 1, odist, CUDA_C, 
        inembed, 1, idist, CUDA_R, 
        nz, &workSize, CUDA_R);
    
    // Allocate temporarily arrays 
    cudaMalloc((void **)&fl, nz*ntheta*nrho*sizeof(real)); 
    cudaMalloc((void **)&flc, nz*(ntheta/2+1)*nrho*sizeof(complex)); 
    cudaMalloc((void **)&gtmp, nz*n*nproj*sizeof(real)); 
    
    // Init texture references
    // texture texfl
    cudaChannelFormatDesc texf_desc;
    texf_desc = CUDA_CREATE_CHANNEL_DESC();     
    cudaExtent volumeSize = make_cudaExtent(ntheta,nrho,nz); 
	
	cudaMalloc3DArray(&fla, &texf_desc,volumeSize,cudaArrayLayered); 
	texfl.addressMode[0] = cudaAddressModeWrap;
	texfl.addressMode[1] = cudaAddressModeWrap;
	texfl.filterMode = cudaFilterModeLinear;
	texfl.normalized  = true;
    cudaBindTextureToArray(texfl, fla,texf_desc); 

    // texture texg
    texf_desc = CUDA_CREATE_CHANNEL_DESC();    
    volumeSize = make_cudaExtent(n,nproj,nz); 
	cudaMalloc3DArray(&ga, &texf_desc, volumeSize,cudaArrayLayered);
	texg.addressMode[0] = cudaAddressModeWrap;
	texg.addressMode[1] = cudaAddressModeWrap;
	texg.filterMode = cudaFilterModeLinear;
	texg.normalized = true;
    cudaBindTextureToArray(texg, ga,texf_desc);
    
    // // texture texg    
    // texf_desc = cudaCreateChannelDesc<real>();	
    // volumeSize = make_cudaExtent(n,nproj,nz);     
    // cudaMalloc3DArray(&ga, &texf_desc, volumeSize,cudaArrayLayered);
    
    // cudaResourceDesc texgRes;
    // memset(&texgRes,0,sizeof(cudaResourceDesc));
    // texgRes.resType            = cudaResourceTypeArray;
    // texgRes.res.array.array    = ga;
    // cudaTextureDesc             texgDescr;    
    // memset(&texgDescr,0,sizeof(cudaTextureDesc));
    
	// texgDescr.addressMode[0] = cudaAddressModeWrap;
	// texgDescr.addressMode[1] = cudaAddressModeWrap;
	// texgDescr.filterMode = cudaFilterModeLinear;
    // texgDescr.normalizedCoords = true;
    // texgDescr.readMode = cudaReadModeNormalizedFloat;
    // cudaArrayLayered
    // cudaCreateTextureObject(&texg, &texgRes, &texgDescr, NULL);
    
    is_free = false;    
}

// destructor, memory deallocation
cfunc::~cfunc() { free(); }

void cfunc::free() {
    if (!is_free) {
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);    
        cudaFree(fl);
        cudaFree(flc);
        is_free = true;
    }
}

void cfunc::setgrids(size_t fz_, size_t lp2p1_, size_t lp2p2_, size_t lp2p1w_, size_t lp2p2w_, 
    size_t C2lp1_, size_t C2lp2_, size_t lpids_, size_t wids_, size_t cids_, 
    size_t nlpids_, size_t nwids_, size_t ncids_){
        
    fz = (complex*)fz_;
    lp2p1 = (float*)lp2p1_;
    lp2p2 = (float*)lp2p2_;
    lp2p1w = (float*)lp2p1w_;
    lp2p2w = (float*)lp2p2w_;
    C2lp1 = (float*)C2lp1_;
    C2lp2 = (float*)C2lp2_;
    lpids = (int*)lpids_;
    wids = (int*)wids_;
    cids = (int*)cids_;
    nlpids = nlpids_;
    nwids = nwids_;
    ncids = ncids_;        
}

void cfunc::backprojection(size_t f_, size_t g_, size_t stream_) 
{
    real* f = (real*)f_;
    real* g = (real*)g_;
    cudaStream_t stream = (cudaStream_t)stream_;
    // set thread block and grid sizes
    dim3 dimBlock(BS1,BS2,BS3);    
    uint GS1, GS2, GS3;    
    
    cufftSetStream(plan_forward, stream);
    cufftSetStream(plan_inverse, stream);    
    cudaMemsetAsync(f, 0, nz*n*n*sizeof(real),stream); 

	//transpose for optimal cache usage
	GS1 = (uint)ceil(n/(float)BS1); GS2 = (uint)ceil(nproj/(float)BS2);GS3 = (uint)ceil(nz/(float)BS3);dim3 dimGrid1(GS1,GS2,GS3);    
	transpose<<<dimGrid1,dimBlock, 0, stream>>>(gtmp, g,n, nproj,nz);
	//compensate in samples for x direction
	GS1 = (uint)ceil(nproj/(float)BS1);GS2 = (uint)ceil(nz/(float)BS2); dim3 dimGrid2(GS1,GS2,1);    	
	SamplesToCoefficients2DY<<<dimGrid2, dimBlock, 0, stream>>>(gtmp,nproj*sizeof(real),nproj, n,nz);
	// //transpose back
	GS1 = (uint)ceil(nproj/(float)BS1);GS2 = (uint)ceil(n/(float)BS2);GS3 = (uint)ceil(nz/(float)BS3); dim3 dimGrid3(GS1,GS2,GS3);    	
	transpose<<<dimGrid3,dimBlock, 0, stream>>>(g,gtmp,nproj, n,nz);
	//compensate in samples for y direction
	GS1 = (uint)ceil(n/(float)BS1);GS2 = (uint)ceil(nz/(float)BS2); dim3 dimGrid4(GS1,GS2,1); 
	SamplesToCoefficients2DY<<<dimGrid4, dimBlock, 0, stream>>>(g,n*sizeof(real),n,nproj,nz);

    copy3DDeviceToArray(ga,g,make_cudaExtent(n, nproj, nz),stream);

    //iterations over log-polar angular spans
    for(int k=0; k<3;k++)
    {
        cudaMemsetAsync(fl, 0, nz*ntheta*nrho*sizeof(real),stream); 
		//interp from polar to log-polar grid
        GS1 = (uint)ceil(ceil(sqrt(nlpids))/(float)BS1); GS2 = (uint)ceil(ceil(sqrt(nlpids))/(float)BS2);GS3 = (uint)ceil(nz/(float)BS3);dim3 dimGrid1(GS1,GS2,GS3);    
        interp<<<dimGrid1, dimBlock, 0, stream>>>(0, fl,&lp2p2[k*nlpids],&lp2p1[k*nlpids],BS1*GS1,nlpids,n,nproj,nz,lpids,ntheta*nrho);
		//interp from polar to log-polar grid additional points
        GS1 = (uint)ceil(ceil(sqrt(nwids))/(float)BS1); GS2 = (uint)ceil(ceil(sqrt(nwids))/(float)BS2);GS3 = (uint)ceil(nz/(float)BS3);dim3 dimGrid2(GS1,GS2,GS3);    
        interp<<<dimGrid2, dimBlock, 0, stream>>>(2, fl,&lp2p2w[k*nwids],&lp2p1w[k*nwids],BS1*GS1,nwids,n,nproj,nz,wids,ntheta*nrho);
        //Forward FFT
        cufftXtExec(plan_forward, fl,flc,CUFFT_FORWARD);        
		//multiplication by adjoint fZ
        GS1 = (uint)ceil((ntheta/2+1)/(float)BS1); GS2 = (uint)ceil(nrho/(float)BS2);GS3 = (uint)ceil(nz/(float)BS3);dim3 dimGrid3(GS1,GS2,GS3);    
        mul<<<dimGrid3, dimBlock, 0, stream>>>(flc,fz,ntheta/2+1,nrho,nz);
		//Inverse FFT
        cufftXtExec(plan_inverse,flc,fl,CUFFT_INVERSE);        
        //copy to binded texture 
        copy3DDeviceToArray(fla,fl,make_cudaExtent(ntheta, nrho, nz),stream);
        //interp from log-polar to Cartesian grid
        GS1 = (uint)ceil(ceil(sqrt(ncids))/(float)BS1); GS2 = (uint)ceil(ceil(sqrt(ncids))/(float)BS2);GS3 = (uint)ceil(nz/(float)BS3);dim3 dimGrid4(GS1,GS2,GS3);
		interp<<<dimGrid4, dimBlock, 0, stream>>>(3, f,&C2lp1[k*ncids],&C2lp2[k*ncids],BS1*GS1,ncids,ntheta,nrho,nz,cids,n*n);                    
    }
}

