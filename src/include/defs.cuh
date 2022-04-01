#define BS1 32
#define BS2 32
#define BS3 1

#ifdef HALF 
    #define CUDA_R CUDA_R_16F
    #define CUDA_C CUDA_C_16F
    #define mexp(x) hexp(x)
    typedef half real;
    typedef half2 real2;
#else
    #define CUDA_R CUDA_R_32F
    #define CUDA_C CUDA_C_32F
    #define mexp(x) __expf(x)
    typedef float real;
    typedef float2 real2;        
#endif
