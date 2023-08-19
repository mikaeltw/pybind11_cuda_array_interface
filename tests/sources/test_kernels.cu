/*
## Copyright (c) 2021, Mikael Twengström
## All rights reserved.
## This file is part of pybind11_cuda_array_interface and is distributed under the
## BSD-3 Clause License. For full terms see the included LICENSE file.
*/

__global__ void saxpy_kernel(float *s_ptr, const float *x_ptr, const float *y_ptr,
                             const int a_scalar, const int length)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x) {
        s_ptr[i] = a_scalar * x_ptr[i] + y_ptr[i];
    }
}

void call_saxpy(float *s_ptr, const float *x_ptr, const float *y_ptr, const int a_scalar,
                const int length)
{
    saxpy_kernel<<<256, 32>>>(s_ptr, x_ptr, y_ptr, a_scalar, length);
}
