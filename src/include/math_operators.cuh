#include<cfunc.cuh>
inline __device__ __host__ complex make_complex(real a, real b)
{
    complex r;
    r.x = a;
    r.y = b;
    return r;
}
// lerp
inline __device__ __host__ real lerp(real a, real b, real t)
{
    return a + t*(b-a);
}

// addition
inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
    return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2 &a, int2 b)
{
    a.x += b.x; a.y += b.y;
}

// subtract
inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
    return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2 &a, int2 b)
{
    a.x -= b.x; a.y -= b.y;
}

// multiply
inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
    return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ int2 operator*(int2 a, int s)
{
    return make_int2(a.x * s, a.y * s);
}
inline __host__ __device__ int2 operator*(int s, int2 a)
{
    return make_int2(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(int2 &a, int s)
{
    a.x *= s; a.y *= s;
}

// complex functions

// additional constructors
inline __host__ __device__ complex make_complex(real s)
{
    return make_complex(s, s);
}
inline __host__ __device__ complex make_complex(int2 a)
{
    return make_complex(real(a.x), real(a.y));
}

// addition
inline __host__ __device__ complex operator+(complex a, complex b)
{
    return make_complex(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ complex operator+(complex a, real b)
{
    return make_complex(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(complex &a, complex b)
{
    a.x += b.x; a.y += b.y;
}

// subtract
inline __host__ __device__ complex operator-(complex a, complex b)
{
    return make_complex(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ complex operator-(real a, complex b)
{
    return make_complex(a - b.x, a - b.y);
}
inline __host__ __device__ complex operator-(complex a, real b)
{
    return make_complex(a.x - b, a.y - b);
}
inline __host__ __device__ void operator-=(complex &a, complex b)
{
    a.x -= b.x; a.y -= b.y;
}

// multiply
inline __host__ __device__ complex operator*(complex a, complex b)
{
    return make_complex(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ complex operator*(complex a, real s)
{
    return make_complex(a.x * s, a.y * s);
}
inline __host__ __device__ complex operator*(real s, complex a)
{
    return make_complex(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(complex &a, real s)
{
    a.x *= s; a.y *= s;
}

// divide
inline __host__ __device__ complex operator/(complex a, complex b)
{
    return make_complex(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ complex operator/(complex a, real s)
{
    real inv = static_cast<real>(1.0f) / s;
    return a * inv;
}
inline __host__ __device__ complex operator/(real s, complex a)  //Danny
{
//    real inv = static_cast<real>(1.0f) / s;
//    return a * inv;
	return make_complex(s / a.x, s / a.y);
}
inline __host__ __device__ void operator/=(complex &a, real s)
{
    real inv = static_cast<real>(1.0f) / s;
    a *= inv;
}

// lerp
inline __device__ __host__ complex lerp(complex a, complex b, real t)
{
    return a + t*(b-a);
}

// dot product
inline __host__ __device__ real dot(complex a, complex b)
{ 
    return a.x * b.x + a.y * b.y;
}

// length
inline __host__ __device__ real length(complex v)
{
    return sqrtf(dot(v, v));
}

// normalize
inline __host__ __device__ complex normalize(complex v)
{
    real invLen = static_cast<real>(1.0f) / sqrtf(dot(v, v));
    return v * invLen;
}

// floor
inline __host__ __device__ complex floor(const complex v)
{
    return make_complex(floor(v.x), floor(v.y));
}


inline __host__ __device__ float3 operator-(real a, float3 b)
{
	return make_float3(a - b.x, a - b.y, a - b.z);
}