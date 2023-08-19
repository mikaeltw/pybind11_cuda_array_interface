/*
## Copyright (c) 2023, Mikael Twengström
## All rights reserved.
## This file is part of pybind11_cuda_array_interface and is distributed under the
## BSD-3 Clause License. For full terms see the included LICENSE file.
*/

#include "pybind11_cuda_array_interface/pybind11_cuda_array_interface.hpp"

#include "pybind11/embed.h"

#include "gtest/gtest.h"

namespace py = pybind11;

class PythonEnvironment : public ::testing::Environment
{
public:
    ~PythonEnvironment() override = default;

    void SetUp() override { py::initialize_interpreter(); }

    void TearDown() override { py::finalize_interpreter(); }
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PythonEnvironment);
    return RUN_ALL_TESTS();
}

inline py::module create_module(const std::string &module_name)
{
    return py::module_::create_extension_module(module_name.c_str(), nullptr,
                                                new py::module_::module_def);
}

namespace cai {

template <typename T>
inline const cuda_array_t<T> &send_and_receive_cuda_array_interface(const cuda_array_t<T> &cai_obj)
{
    return cai_obj;
}

class CudaArrayInterfaceTest : public ::testing::Test
{
protected:
    // The test class is declared as a friend inside cuda_array_t<T> and cuda_memory_handle<T>.

    void SetUp() override {}

    void TearDown() override {}

    template <typename T>
    std::shared_ptr<cuda_memory_handle<T>>
    create_shared_cuda_memory_handle(T *ptr, std::function<void(void *)> deleter)
    {
        return cuda_memory_handle<T>::make_shared_handle(ptr, deleter);
    }

    template <typename T>
    std::shared_ptr<cuda_memory_handle<T>> create_shared_cuda_memory_handle(T *ptr)
    {
        return cuda_memory_handle<T>::make_shared_handle(ptr);
    }

    template <typename T>
    cai::cuda_shared_ptr_holder<T> *
    create_cuda_shared_ptr_holder(std::shared_ptr<cuda_memory_handle<T>> handle)
    {
        return cai::cuda_shared_ptr_holder<T>::create(handle);
    }
};

} // namespace cai

using cai::CudaArrayInterfaceTest;

TEST(ValidateTypedPointerTest, CompatibleTypeNonReadOnly)
{
    auto arr = cai::cuda_array_t<float>({1});
    EXPECT_NO_THROW(arr.get_compatible_typed_pointer());
}

TEST(ValidateTypedPointerTest, CompatibleTypeReadOnly)
{
    auto arr = cai::cuda_array_t<float>({1}, true);
    EXPECT_THROW(arr.get_compatible_typed_pointer(), caiexcp::ReadOnlyAccessError);
}

TEST(ValidateTypedPointerTest, InCompatibleTypeReadOnly)
{
    const auto arr = cai::cuda_array_t<float>({1});
    EXPECT_NO_THROW(arr.get_compatible_typed_pointer());
}

TEST(ValidateTypedPointerTest, InCompatibleTypeReadOnlySaveConst)
{
    const auto arr = cai::cuda_array_t<float>({1}, true);
    EXPECT_NO_THROW(arr.get_compatible_typed_pointer());
}

TEST(ValidateTypedPointerTest, DifferentDataTypeTest)
{
    auto arr = cai::cuda_array_t<int>({1});
    EXPECT_NO_THROW(arr.get_compatible_typed_pointer());
}

TEST(ValidateTypedPointerTest, MemoryAllocationAndDeallocation)
{
    EXPECT_NO_THROW({ cai::cuda_array_t<int> arr({1}); });

    {
        cai::cuda_array_t<int> arr({1});
    }

    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(CudaArrayMethodsTest, ConstructorTest)
{
    auto arr = cai::cuda_array_t<float>({1, 2, 3});
    EXPECT_EQ(arr.get_shape().size(), 3);
}

TEST(CudaArrayMethodsTest, ReadOnlyTrue)
{
    auto arr = cai::cuda_array_t<float>({1}, true);
    EXPECT_TRUE(arr.is_readonly());
}

TEST(CudaArrayMethodsTest, ReadOnlyFalse)
{
    auto arr = cai::cuda_array_t<float>({1}, false);
    EXPECT_FALSE(arr.is_readonly());
}

TEST(CudaArrayMethodsTest, VersionTest)
{
    auto arr = cai::cuda_array_t<float>({1}, false, 4);
    EXPECT_EQ(arr.get_version(), 4);
}

TEST(CudaArrayMethodsTest, SizeOfShapeTest)
{
    auto arr = cai::cuda_array_t<float>({2, 3, 4});
    EXPECT_EQ(arr.size_of_shape(), 24);
}

TEST(CudaArrayMethodsTest, DataTypeTest)
{
    auto arr = cai::cuda_array_t<float>({1});
    EXPECT_EQ(arr.get_dtype().kind(), 'f');
}

TEST(ValidateTypestr, TooShort)
{
    EXPECT_THROW(cai::validate_typestr(""), caiexcp::InvalidTypestrError);
    EXPECT_THROW(cai::validate_typestr("<"), caiexcp::InvalidTypestrError);
    EXPECT_THROW(cai::validate_typestr("<t"), caiexcp::InvalidTypestrError);
}

TEST(ValidateTypestr, InvalidEndianness)
{
    EXPECT_THROW(cai::validate_typestr("xt4"), caiexcp::InvalidTypestrError);
    EXPECT_THROW(cai::validate_typestr("at4"), caiexcp::InvalidTypestrError);
}

TEST(ValidateTypestr, InvalidTypeCharacterCode)
{
    EXPECT_THROW(cai::validate_typestr("<x4"), caiexcp::InvalidTypestrError);
    EXPECT_THROW(cai::validate_typestr(">y8"), caiexcp::InvalidTypestrError);
}

TEST(ValidateTypestr, InvalidByteSize)
{
    EXPECT_THROW(cai::validate_typestr("<ta"), caiexcp::InvalidTypestrError);
    EXPECT_THROW(cai::validate_typestr(">tb"), caiexcp::InvalidTypestrError);
}

TEST(ValidateTypestr, ZeroByteSize)
{
    EXPECT_THROW(cai::validate_typestr("<t0"), caiexcp::InvalidTypestrError);
    EXPECT_THROW(cai::validate_typestr(">i0"), caiexcp::InvalidTypestrError);
}

TEST(ValidateTypestr, ValidTypestr)
{
    EXPECT_NO_THROW(cai::validate_typestr("<t4"));
    EXPECT_NO_THROW(cai::validate_typestr(">i8"));
    EXPECT_NO_THROW(cai::validate_typestr("|u16"));
}

TEST(ValidateShape, IsZero)
{
    std::vector<size_t> shape = {};

    EXPECT_THROW(cai::validate_shape(shape), caiexcp::InvalidShapeError);
}

TEST(ValidateShape, ContainsZero)
{
    std::vector<size_t> shape1 = {5, 3, 0};
    std::vector<size_t> shape2 = {0, 6, 7, 8};

    EXPECT_THROW(cai::validate_shape(shape1), caiexcp::InvalidShapeError);
    EXPECT_THROW(cai::validate_shape(shape2), caiexcp::InvalidShapeError);
}

TEST(ValidateShape, ValidShape)
{
    std::vector<size_t> shape1 = {5, 3, 4};
    std::vector<size_t> shape2 = {2, 6, 7, 8};

    EXPECT_NO_THROW(cai::validate_shape(shape1));
    EXPECT_NO_THROW(cai::validate_shape(shape2));
}

TEST(ValidateCudaPtr, ValidCudaPointer)
{
    int *devPtr;
    cudaMalloc(reinterpret_cast<void **>(&devPtr), sizeof(int));

    EXPECT_NO_THROW(cai::validate_cuda_ptr(reinterpret_cast<void *>(devPtr)));

    cudaFree(devPtr);
}

TEST(ValidateCudaPtr, InvalidCudaPointer)
{
    int localVariable;

    EXPECT_THROW(cai::validate_cuda_ptr(reinterpret_cast<void *>(&localVariable)),
                 caiexcp::UnRegCudaTypeError);
    EXPECT_THROW(cai::validate_cuda_ptr(nullptr), caiexcp::UnRegCudaTypeError);
}

TEST(ValidateCapsule, InvalidCapsule)
{
    py::capsule invalidCapsule;

    EXPECT_THROW(cai::validate_capsule(invalidCapsule), caiexcp::InvalidCapsuleError);
}

TEST(ValidateCapsule, UnexpectedCapsuleName)
{
    int data = 42;
    py::capsule namedCapsule(&data, "unexpected_name");

    EXPECT_THROW(cai::validate_capsule(namedCapsule), caiexcp::InvalidCapsuleError);
}

TEST(ValidateCapsule, ValidCapsule)
{
    int data = 42;
    py::capsule fullyValidCapsule(&data, "cuda_memory_capsule");

    EXPECT_NO_THROW(cai::validate_capsule(fullyValidCapsule));
}

TEST_F(CudaArrayInterfaceTest, ValidateCudaMemoryHandleInvalidRefCount)
{
    // Allocate CUDA memory
    int *devPtr;
    cudaMalloc(reinterpret_cast<void **>(&devPtr), sizeof(int));

    auto handle1 = create_shared_cuda_memory_handle<int>(devPtr);
    auto ___ = handle1;     // Increases the reference count
    static_cast<void>(___); // Variable unused

    EXPECT_THROW(cai::validate_cuda_memory_handle(handle1), caiexcp::ObjectOwnershipError);
}

TEST_F(CudaArrayInterfaceTest, ValidateCudaMemoryHandleValidHandle)
{
    int *devPtr;
    cudaMalloc(reinterpret_cast<void **>(&devPtr), sizeof(int));

    auto handle = create_shared_cuda_memory_handle<int>(devPtr);

    EXPECT_NO_THROW(cai::validate_cuda_memory_handle(handle));
}

TEST_F(CudaArrayInterfaceTest, CudaSharedPtrHolderCorrectInstantiation)
{
    int *devPtr;
    cudaMalloc(reinterpret_cast<void **>(&devPtr), sizeof(int));

    auto handle = create_shared_cuda_memory_handle<int>(devPtr);
    auto *holder = create_cuda_shared_ptr_holder<int>(handle);

    EXPECT_NE(holder, nullptr);

    delete holder;
}

TEST(CudaArraySimulatedIntegrationTest, SendAndReceive)
{

    auto test_module = create_module("test");

    test_module.def("sendandreceive", &cai::send_and_receive_cuda_array_interface<int>);

    py::module cupy = py::module::import("cupy");
    py::module numpy = py::module::import("numpy");

    py::list lst = py::cast(std::vector<int>({1, 2, 3, 4, 5}));
    py::object cupy_array = cupy.attr("array")(lst).attr("astype")("int32");

    py::object received_cupy_array = test_module.attr("sendandreceive")(cupy_array);

    // Check if the returned object has __cuda_array_interface__
    ASSERT_TRUE(py::hasattr(received_cupy_array, "__cuda_array_interface__"));

    auto received_numpy_array = cupy.attr("asnumpy")(received_cupy_array).cast<py::array_t<int>>();
    auto sent_numpy_array = cupy.attr("asnumpy")(cupy_array).cast<py::array_t<int>>();

    ASSERT_TRUE(numpy.attr("array_equal")(received_numpy_array, sent_numpy_array).cast<bool>());
}
