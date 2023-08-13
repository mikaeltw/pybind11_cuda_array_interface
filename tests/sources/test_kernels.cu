/*
## Copyright (c) 2021, Mikael Twengström
## All rights reserved.
## This file is part of pybind11_cuda_array_interface and is distributed under the
## BSD-3 Clause License. For full terms see the included LICENSE file.
*/

__global__
void saxpy_kernel(float *s, float *x, float *y, int a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
         s[i] = a * x[i] + y[i];
    }
}

void call_saxpy(float *s, float *x, float *y, int a, int n) {
    saxpy_kernel<<<256, 32>>>(s, x, y, a, n);
}