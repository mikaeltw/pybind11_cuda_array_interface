#include "../../include/pybind11_cuda_array_interface/pybind11_cuda_array_interface.h"

#include <iostream>

void send_cupy(pybind11::cuda_array) {
    std::cout << "Hello" << std::endl;
}

PYBIND11_MODULE(pybind11_cuda_array_interface_test, module) {
    module.doc() = "Demagnetising factor CUDA library";

    module.def("send_cupy", &send_cupy);
}
