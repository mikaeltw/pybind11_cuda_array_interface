/*
## Copyright (c) 2021, Mikael Twengström
## All rights reserved.
## This file is part of pybind11_cuda_array_interface and is distributed under the
## BSD-3 Clause License. For full terms see the included LICENSE file.
*/

#pragma once

#include <cuda_runtime.h>

__global__ void saxpy_kernel(float *s, float *x, float *y, int a, int n);

void call_saxpy(float *s, float *x, float *y, int a, int n);
