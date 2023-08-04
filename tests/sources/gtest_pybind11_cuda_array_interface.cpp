#include "pybind11_cuda_array_interface/pybind11_cuda_array_interface.hpp"

#include "pybind11/embed.h"

#include "gtest/gtest.h"


namespace py = pybind11;

class PythonEnvironment : public ::testing::Environment {
    public:
        ~PythonEnvironment() override = default;

        void SetUp() override {
            py::initialize_interpreter();
        }

        void TearDown() override {
            py::finalize_interpreter();
        }
};

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PythonEnvironment);
    return RUN_ALL_TESTS();
}

inline py::module create_module(const std::string& module_name) {
    return py::module_::create_extension_module(module_name.c_str(), nullptr, new py::module_::module_def);
}

namespace cai {

    template <typename T>
    inline const cuda_array_t<T>& send_and_receive_cuda_array_interface(const cuda_array_t<T>& ca)
    {
        return ca;
    }

    class CudaArrayInterfaceTest : public ::testing::Test {
        protected:
            // The test class is declared as a friend inside cuda_array_t<T> and cuda_memory_handle<T>.
            void* deviceptr;

            void SetUp() override {}

            void TearDown() override {}

            void allocate_deviceptr(size_t size) {
                checkCudaErrors(cudaMalloc(&deviceptr, size));
            }

            template <typename T>
            cuda_array_t<T> create_cuda_array(bool readonly = false) {
                allocate_deviceptr(1024 * sizeof(T));
                cuda_array_t<T> arr;
                arr.handle = cuda_memory_handle<T>::make_shared_handle(deviceptr);
                arr.readonly = readonly;
                arr.typestr = py::format_descriptor<T>::format();
                arr.shape = {5};
                arr.version = 3;
                return arr;
            }
    };

}

using cai::CudaArrayInterfaceTest;

TEST_F(CudaArrayInterfaceTest, CompatibleTypeNonReadOnly) {
    auto arr = create_cuda_array<float>();
    EXPECT_NO_THROW(arr.get_compatible_typed_pointer());
}

TEST_F(CudaArrayInterfaceTest, CompatibleTypeReadOnly) {
    auto arr = create_cuda_array<float>(true);
    EXPECT_THROW(arr.get_compatible_typed_pointer(), std::runtime_error);
}

TEST_F(CudaArrayInterfaceTest, InCompatibleTypeReadOnly) {
    const auto arr = create_cuda_array<float>();
    EXPECT_NO_THROW(arr.get_compatible_typed_pointer());
}

TEST_F(CudaArrayInterfaceTest, InCompatibleTypeReadOnlySaveConst) {
    const auto arr = create_cuda_array<float>(true);
    EXPECT_NO_THROW(arr.get_compatible_typed_pointer());
}

TEST(CudaArrayInterface, SendAndReceive) {

    auto m = create_module("test");

    m.def("sendandreceive", &cai::send_and_receive_cuda_array_interface<float>);

    py::module cp = py::module::import("cupy");
    py::module np = py::module::import("numpy");

    py::list lst = py::cast(std::vector<int>({1, 2, 3, 4, 5}));
    py::object cupy_array = cp.attr("array")(lst).attr("astype")("int32");

    py::object received_cupy_array = m.attr("sendandreceive")(cupy_array);

    // Check if the returned object has __cuda_array_interface__
    ASSERT_TRUE(py::hasattr(received_cupy_array, "__cuda_array_interface__"));

    py::array_t<int> received_numpy_array = cp.attr("asnumpy")(received_cupy_array).cast<py::array_t<int>>();
    py::array_t<int> sent_numpy_array = cp.attr("asnumpy")(cupy_array).cast<py::array_t<int>>();

    ASSERT_TRUE(np.attr("array_equal")(received_numpy_array, sent_numpy_array).cast<bool>());
}
