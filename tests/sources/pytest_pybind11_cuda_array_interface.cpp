#define PYBIND11_DETAILED_ERROR_MESSAGES
#include "pybind11_cuda_array_interface/pybind11_cuda_array_interface.hpp"

#include "test_kernels.hpp"

#include <iostream>

void saxpy(cai::cuda_array_t s, cai::cuda_array_t x, cai::cuda_array_t y, int a) {

    auto s_ptr = s.get_compatible_typed_pointer<float>();
    auto x_ptr = x.get_compatible_typed_pointer<float>();
    auto y_ptr = y.get_compatible_typed_pointer<float>();

    call_saxpy(s_ptr, x_ptr, y_ptr, a, s.size_of_shape());
}

cai::cuda_array_t receive_and_return_cuda_array_interface(cai::cuda_array_t s) {
    return s;
}

PYBIND11_MODULE(pycai, module) {
    module.doc() = "Test module for __cuda_array_interface__";

    module.def("saxpy", &saxpy);
    module.def("receive_and_return_cuda_array_interface", &receive_and_return_cuda_array_interface);

}

//m.def("create_cuda_array", &create_cuda_array, py::return_value_policy::take_ownership, py::keep_alive<0, 1>());
