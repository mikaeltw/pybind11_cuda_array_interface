#pragma once

#include <cuda_runtime.h>

__global__
void saxpy_kernel(float *s, float *x, float *y, int a, int n);

void call_saxpy(float *s, float *x, float *y, int a, int n);
