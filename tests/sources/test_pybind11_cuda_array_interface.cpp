#include "../../include/pybind11_cuda_array_interface/pybind11_cuda_array_interface.h"

#include "test_kernels.h"

#include <iostream>

void saxpy(pybind11::cuda_array s, pybind11::cuda_array x, pybind11::cuda_array y, int a) {
    auto s_info = s.get_cuda_array_interface();
    auto x_info = x.get_cuda_array_interface();
    auto y_info = y.get_cuda_array_interface();

    float *s_ptr = reinterpret_cast<float *>(s_info.device_ptr);
    float *x_ptr = reinterpret_cast<float *>(x_info.device_ptr);
    float *y_ptr = reinterpret_cast<float *>(y_info.device_ptr);

    call_saxpy(s_ptr, x_ptr, y_ptr, a, s_info.size);
}

PYBIND11_MODULE(pybind11_cuda_array_interface_test, module) {
    module.doc() = "Test module for __cuda_array_interface__";

    module.def("saxpy", &saxpy);

}
