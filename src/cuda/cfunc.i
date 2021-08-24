/*interface*/
%module cfunc

%{
#define SWIG_FILE_WITH_INIT
#include "cfunc.cuh"
%}

class cfunc
{
public:
    %immutable;
    int n;      
    int nproj;      
    int ntheta; 
    int nrho; 
    int nz;  
    
    %mutable;
    cfunc(size_t nproj, size_t nz, size_t n, size_t ntheta, size_t nrho);
	~cfunc();      
    void free();
    void setgrids(size_t fz, size_t lp2p1, size_t lp2p2, size_t lp2p1w, size_t lp2p2w, 
        size_t C2lp1, size_t C2lp2, size_t lpids, size_t wids, size_t cids, 
        size_t nlpids, size_t nwids, size_t ncids);	
    void backprojection(size_t f, size_t g, size_t stream);
};