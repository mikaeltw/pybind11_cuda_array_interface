#include "../include/pybind11_cuda_array_interface/pybind11_cuda_array_interface.h"

#include <stdio.h>

void send_cupy(pybind11::cuda_array) {
    std::cout << "Hello" << endl;
}

PYBIND11_MODULE(test_cai, module) {
    module.doc() = "Demagnetising factor CUDA library";

    module.def("Send_cupy", &send_cupy);
}
