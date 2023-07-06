#include "pybind11_cuda_array_interface/pybind11_cuda_array_interface.hpp"

#include "pybind11/embed.h"

#include "gtest/gtest.h"


namespace py = pybind11;

inline py::module create_module(const std::string& module_name) {
    return py::module_::create_extension_module(module_name.c_str(), nullptr, new py::module_::module_def);
}

inline const cai::cuda_array_t& send_and_receive_cuda_array_interface(const cai::cuda_array_t& ca)
{
    return ca;
}

TEST(CudaArrayInterface, SendAndReceive) {
    py::scoped_interpreter guard;

    auto m = create_module("test");

    m.def("sendandreceive", &send_and_receive_cuda_array_interface);

    py::module cp = py::module::import("cupy");
    py::module np = py::module::import("numpy");

    py::list lst = py::cast(std::vector<int>({1, 2, 3, 4, 5}));
    py::object cupy_array = cp.attr("array")(lst).attr("astype")("int32");

    py::object result = m.attr("sendandreceive")(cupy_array);

    // Check if the returned object has __cuda_array_interface__
    ASSERT_TRUE(py::hasattr(result, "__cuda_array_interface__"));

    py::object received_cupy_array = cp.attr("asarray")(result);

    py::array_t<int> received_numpy_array = cp.attr("asnumpy")(received_cupy_array).cast<py::array_t<int>>();
    py::array_t<int> sent_numpy_array = cp.attr("asnumpy")(cupy_array).cast<py::array_t<int>>();

    ASSERT_TRUE(np.attr("array_equal")(received_numpy_array, sent_numpy_array).cast<bool>());
}