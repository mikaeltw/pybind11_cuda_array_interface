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

    inline const cuda_array_t& send_and_receive_cuda_array_interface(const cuda_array_t& ca)
    {
        return ca;
    }

    class CudaArrayInterfaceTest : public ::testing::Test {
        protected:
            // The test class is declared as a friend inside cuda_array_t and cuda_memory_handle.
            CUdevice device;
            CUcontext context;
            CUdeviceptr deviceptr;

            void SetUp() override {
                checkCudaErrors(cuInit(0));
                checkCudaErrors(cuDeviceGet(&device, 0));
                checkCudaErrors(cuCtxCreate(&context, 0, device));
            }

            void TearDown() override {
                checkCudaErrors(cuCtxDestroy(context));
            }

            void allocate_deviceptr(size_t size) {
                checkCudaErrors(cuMemAlloc(&deviceptr, size));
            }

            cuda_array_t create_cuda_array(bool readonly = false) {
                allocate_deviceptr(1024 * sizeof(float));
                cuda_array_t arr;
                arr.handle = cuda_memory_handle::make_shared_handle(deviceptr);
                arr.readonly = readonly;
                arr.typestr = "<f4";  // Assuming float corresponds to "<f4"
                arr.shape = {5};
                arr.version = 3;
                return arr;
            }
    };

}

using cai::CudaArrayInterfaceTest;

TEST_F(CudaArrayInterfaceTest, CompatibleTypeNonReadOnly) {
    cai::cuda_array_t arr = create_cuda_array();
    EXPECT_NO_THROW(arr.get_compatible_typed_pointer<float>());
}

TEST_F(CudaArrayInterfaceTest, CompatibleTypeReadOnly) {
    cai::cuda_array_t arr = create_cuda_array(true);
    EXPECT_THROW(arr.get_compatible_typed_pointer<float>(), std::runtime_error);
}

TEST_F(CudaArrayInterfaceTest, InCompatibleTypeReadOnly) {
    const cai::cuda_array_t arr = create_cuda_array();
    EXPECT_NO_THROW(arr.get_compatible_typed_pointer<float>());
}

TEST_F(CudaArrayInterfaceTest, InCompatibleTypeReadOnlySaveConst) {
    const cai::cuda_array_t arr = create_cuda_array(true);
    EXPECT_NO_THROW(arr.get_compatible_typed_pointer<float>());
}

TEST_F(CudaArrayInterfaceTest, IncompatibleType) {
    cai::cuda_array_t arr = create_cuda_array();
    EXPECT_THROW(arr.get_compatible_typed_pointer<int>(), std::runtime_error);
}

TEST(CudaArrayInterface, SendAndReceive) {

    auto m = create_module("test");

    m.def("sendandreceive", &cai::send_and_receive_cuda_array_interface);

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
