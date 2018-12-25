#ifndef HALF_VECTOR_TYPES
#define HALF_VECTOR_TYPES
#include "newhalf.hpp"
#include <vector_types.h>
struct half3
{
    half x, y, z;
};

typedef struct half3 half3;

struct half2_3
{
    half2 x, y, z;
};

typedef struct half2_3 half2_3;

struct half3_host
{
    half_float::half x, y, z;
};

typedef struct half3_host half3_host;

struct half2_3_host
{
    uint32_t x, y, z;
};

typedef struct half2_3_host half2_3_host;

#endif
