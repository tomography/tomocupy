#ifndef CFUNC_CUH
#define CFUNC_CUH

#include <cufft.h>

class cfunc
{
    bool is_free;
	float* fl;
    float2* flc;
    float2* fz;
    float* lp2p1;
    float* lp2p2;
    float* lp2p1w;
    float* lp2p2w;
    float* C2lp1;
    float* C2lp2;
    int* lpids;
    int* wids;
    int* cids;
    int nlpids;
    int nwids;
    int ncids;
    cufftHandle plan_forward;
	cufftHandle plan_inverse;

public:
    int n;      
    int nproj;      
    int ntheta; 
    int nrho; 
    int nz;  

    cfunc(int nproj, int nz, int n, int nrho, int ntheta);
	~cfunc();      
    void free();
    void setgrids(size_t fz, size_t lp2p1, size_t lp2p2, size_t lp2p1w, size_t lp2p2w, 
        size_t C2lp1, size_t C2lp2, size_t lpids, size_t wids, size_t cids, 
        size_t nlpids, size_t nwids, size_t ncids);	
    void backprojection(size_t f, size_t g);
};

#endif