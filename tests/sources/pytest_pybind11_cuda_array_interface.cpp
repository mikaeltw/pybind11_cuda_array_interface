#define PYBIND11_DETAILED_ERROR_MESSAGES
#include "pybind11_cuda_array_interface/pybind11_cuda_array_interface.hpp"

#include "test_kernels.hpp"

#include <iostream>


template <typename T>
void saxpy(cai::cuda_array_t<T> s, cai::cuda_array_t<T> x, cai::cuda_array_t<T> y, int a) {

    auto s_ptr = s.get_compatible_typed_pointer();
    auto x_ptr = x.get_compatible_typed_pointer();
    auto y_ptr = y.get_compatible_typed_pointer();

    call_saxpy(s_ptr, x_ptr, y_ptr, a, static_cast<int>(s.size_of_shape()));
}

template <typename T>
cai::cuda_array_t<T> receive_and_return_cuda_array_interface(cai::cuda_array_t<T> s) {
    return s;
}

template <typename T>
cai::cuda_array_t<T> return_cuda_array_interface() {
    cai::cuda_array_t<T> s({5});
    return s;
}

PYBIND11_MODULE(pycai, module) {
    module.doc() = "Test module for __cuda_array_interface__";

    module.def("saxpy", &saxpy<float>);
    module.def("receive_and_return_cuda_array_interface", &receive_and_return_cuda_array_interface<float>);
    module.def("return_cuda_array_interface", &return_cuda_array_interface<float>);

}

//m.def("create_cuda_array", &create_cuda_array, py::return_value_policy::take_ownership, py::keep_alive<0, 1>());

