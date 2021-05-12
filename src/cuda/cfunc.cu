#include "cfunc.cuh"
#include "kernels.cuh"
#include <stdio.h>
cfunc::cfunc(int nproj, int nz, int n, int nrho, int ntheta):
nproj(nproj), nz(nz), n(n), nrho(nrho), ntheta(ntheta) {
    
    int ffts[] = {nrho,ntheta};
	int idist = nrho*ntheta;int odist = nrho*(ntheta/2+1);
	int inembed[] = {nrho, ntheta};int onembed[] = {nrho, ntheta/2+1};
	cufftPlanMany(&plan_forward, 2, ffts, inembed, 1, idist, onembed, 1, odist, CUFFT_R2C, nz); 
	cufftPlanMany(&plan_inverse, 2, ffts, onembed, 1, odist, inembed, 1, idist, CUFFT_C2R, nz); 

    cudaMalloc((void **)&fl, nz*ntheta*nrho*sizeof(float)); 
	cudaMalloc((void **)&flc, nz*(ntheta/2+1)*nrho*sizeof(float2)); 
    is_free = false;    
}

// destructor, memory deallocation
cfunc::~cfunc() { free(); }

void cfunc::setgrids(size_t fz_, size_t lp2p1_, size_t lp2p2_, size_t lp2p1w_, size_t lp2p2w_, 
    size_t C2lp1_, size_t C2lp2_, size_t lpids_, size_t wids_, size_t cids_, 
    size_t nlpids_, size_t nwids_, size_t ncids_){
        
    fz = (float2*)fz_;
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

void cfunc::free() {
    if (!is_free) {
        cufftDestroy(plan_forward);
        cufftDestroy(plan_inverse);    
        cudaFree(fl);
        cudaFree(flc);
        is_free = true;
    }
}

void cfunc::backprojection(size_t f_, size_t g_) 
{
    float* f = (float*)f_;
    float* g = (float*)g_;

    uint BS1 = 32; uint BS2 = 32; uint BS3 = 1;    
	uint GS1, GS2, GS3;    
    dim3 dimBlock(BS1,BS2,BS3);

    for(int k=0; k<3;k++)
    {
        cudaMemset(fl, 0, nz*ntheta*nrho*sizeof(float)); 
		//interp from polar to log-polar grid

        GS1 = (uint)ceil(ceil(sqrt(nlpids))/(float)BS1); GS2 = (uint)ceil(ceil(sqrt(nlpids))/(float)BS2);GS3 = (uint)ceil(nz/(float)BS3);dim3 dimGrid1(GS1,GS2,GS3);
		interplp<<<dimGrid1, dimBlock>>>(fl,g,&lp2p2[k*nlpids],&lp2p1[k*nlpids],BS1*GS1,nlpids,n,nproj,nz,lpids,ntheta*nrho);
		//interp from polar to log-polar grid additional points
        GS1 = (uint)ceil(ceil(sqrt(nwids))/(float)BS1); GS2 = (uint)ceil(ceil(sqrt(nwids))/(float)BS2);GS3 = (uint)ceil(nz/(float)BS3);dim3 dimGrid2(GS1,GS2,GS3);
		interplp<<<dimGrid2, dimBlock>>>(fl,g,&lp2p2w[k*nwids],&lp2p1w[k*nwids],BS1*GS1,nwids,n,nproj,nz,wids,ntheta*nrho);

		//Forward FFT
		cufftExecR2C(plan_forward, (cufftReal*)fl,(cufftComplex*)flc);

		//multiplication by adjoint fZ
		GS1 = (uint)ceil((ntheta/2+1)/(float)BS1); GS2 = (uint)ceil(nrho/(float)BS2);GS3 = (uint)ceil(nz/(float)BS3);dim3 dimGrid3(GS1,GS2,GS3);
		mul<<<dimGrid3, dimBlock>>>(1/(float)(ntheta*nrho),flc,fz,ntheta/2+1,nrho,nz);

		//Inverse FFT
		cufftExecC2R(plan_inverse,(cufftComplex*)flc,(cufftReal*)fl);

		//interp from log-polar to Cartesian grid
		GS1 = (uint)ceil(ceil(sqrt(ncids))/(float)BS1); GS2 = (uint)ceil(ceil(sqrt(ncids))/(float)BS2);GS3 = (uint)ceil(nz/(float)BS3);dim3 dimGrid4(GS1,GS2,GS3);
		interpc<<<dimGrid4, dimBlock>>>(f,fl,&C2lp1[k*ncids],&C2lp2[k*ncids],BS1*GS1,ncids,ntheta,nrho,nz,cids,n*n);            
        
    }
}
