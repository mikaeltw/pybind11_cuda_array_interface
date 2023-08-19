/*
## Copyright (c) 2023, Mikael Twengström
## All rights reserved.
## This file is part of pybind11_cuda_array_interface and is distributed under the
## BSD-3 Clause License. For full terms see the included LICENSE file.
*/

#define PYBIND11_DETAILED_ERROR_MESSAGES
#include "pybind11_cuda_array_interface/pybind11_cuda_array_interface.hpp"

#include "test_kernels.hpp"

#include <iostream>

template <typename T>
void saxpy(cai::cuda_array_t<T> s_cai, cai::cuda_array_t<T> x_cai, cai::cuda_array_t<T> y_cai,
           int a_scalar)
{
    auto s_ptr = s_cai.get_compatible_typed_pointer();
    auto x_ptr = x_cai.get_compatible_typed_pointer();
    auto y_ptr = y_cai.get_compatible_typed_pointer();

    call_saxpy(s_ptr, x_ptr, y_ptr, a_scalar, static_cast<int>(s_cai.size_of_shape()));
}

template <typename T>
cai::cuda_array_t<T> receive_and_return_cuda_array_interface(cai::cuda_array_t<T> s_cai)
{
    return s_cai;
}

template <typename T>
cai::cuda_array_t<T> return_cuda_array_interface(std::vector<size_t> shape = {3, 4, 5},
                                                 bool readonly = false, int version = 3)
{

    cai::cuda_array_t<T> s_cai(shape, readonly, version);
    return s_cai;
}

template <typename T>
cai::cuda_array_t<T>
return_cuda_array_interface_with_memory_copy(std::vector<T> vec = {1.0, 2.0, 3.0, 4.0, 5.0},
                                             bool readonly = false, int version = 3)
{

    cai::cuda_array_t<T> s_cai({vec.size()}, readonly, version);
    checkCudaErrors(cudaMemcpy(s_cai.get_compatible_typed_pointer(), vec.data(),
                               vec.size() * sizeof(T), cudaMemcpyHostToDevice));

    return s_cai;
}

PYBIND11_MODULE(pycai, module)
{
    module.doc() = "Test module for __cuda_array_interface__";
    caiexcp::register_custom_cuda_array_interface_exceptions(module);

    module.def("saxpy", &saxpy<float>,
               "Function performing s = a * x + y, result is returned implicitly in s",
               py::arg("s"), py::arg("x"), py::arg("y"), py::arg("a"));

    module.def("receive_and_return_cuda_array_interface_float",
               &receive_and_return_cuda_array_interface<float>,
               "Function accepts a cuda_array_t and immediately just sends it back", py::arg("s"));

    module.def("receive_and_return_cuda_array_interface_int",
               &receive_and_return_cuda_array_interface<int>,
               "Function accepts a cuda_array_t and immediately just sends it back", py::arg("s"));

    module.def(
        "return_cuda_array_interface", &return_cuda_array_interface<float>,
        "Constructs a cuda_array_t in C++ and returns it to Python",
        py::arg_v("shape", std::vector<size_t>({3, 4, 5}), "std::vector<size_t>({3, 4, 5})"),
        py::arg("readonly") = false, py::arg("version") = 3);

    module.def(
        "return_cuda_array_interface_with_memory_copy",
        &return_cuda_array_interface_with_memory_copy<float>,
        "Constructs a cuda_array_t in C++, copies memory to the GPU and returns it to Python",
        py::arg_v("vec", std::vector<float>({1.0, 2.0, 3.0, 4.0, 5.0}),
                  "std::vector<float>({1.0, 2.0, 3.0, 4.0, 5.0})"),
        py::arg("readonly") = false, py::arg("version") = 3);
}
