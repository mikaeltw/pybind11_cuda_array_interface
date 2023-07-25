import cupy as cp
import numpy as np
import pycai


def test_saxpy_kernel():
    s1 = cp.empty(shape=(3,), dtype=cp.float32)
    s2 = cp.empty(shape=(3,), dtype=cp.float32)
    s3 = cp.array([41.0, 62.0, 98.0], dtype=cp.float32)
    x = cp.array([2.0, 3.0, 5.0], dtype=cp.float32)
    y = cp.array([7.0, 11.0, 13.0], dtype=cp.float32)
    a = 17

    pycai.saxpy(s1, x, y, a)
    s2 = a * x + y

    cp.testing.assert_allclose(s1, s2)
    cp.testing.assert_allclose(s1, s3)


def test_sendreceive():
    cupy_test_array = cp.array([41.0, 62.0, 98.0], dtype=cp.float32)
    numpy_test_array = cp.asnumpy(cupy_test_array)
    returned_cupy_array = pycai.receive_and_return_cuda_array_interface(cupy_test_array)

    assert hasattr(returned_cupy_array, "__cuda_array_interface__")

    np.testing.assert_allclose(numpy_test_array, cp.asnumpy(returned_cupy_array))
