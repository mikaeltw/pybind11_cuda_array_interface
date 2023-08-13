"""
## Copyright (c) 2023, Mikael Twengström
## All rights reserved.
## This file is part of pybind11_cuda_array_interface and is distributed under the
## BSD-3 Clause License. For full terms see the included LICENSE file.
"""

from types import SimpleNamespace
from typing import Any, Dict, Protocol, Tuple

import cupy as cp
import numpy as np
import pycai
import pytest
from pycai import (
    DtypeMismatchError,
    IncompleteInterfaceError,
    InterfaceNotImplementedError,
    InvalidShapeError,
    InvalidTypestrError,
    InvalidVersionError,
    ReadOnlyAccessError,
    UnRegCudaTypeError,
)


class ArrayCapsuleWrapperProtocol(Protocol):
    def __init__(self, array: cp.ndarray, capsule: Any) -> None:
        ...

    @property
    def array(self) -> cp.ndarray:
        ...

    @property
    def capsule(self) -> Any:  # expected to be py::capsule from pybind11
        ...


def cupy_asarray_with_capsule(obj: Any) -> ArrayCapsuleWrapperProtocol:
    # Check if the object has both __cuda_array_interface__ and _capsule attributes
    if not (hasattr(obj, "__cuda_array_interface__") and hasattr(obj, "_capsule")):
        raise ValueError("Input object must have both __cuda_array_interface__ and _capsule attributes.")

    # Convert the object to a CuPy array
    array = cp.asarray(obj)

    class ArrayCapsuleWrapper:
        def __init__(self, array: cp.ndarray, capsule: Any) -> None:
            self._array = array
            self._capsule = capsule

        @property
        def array(self) -> cp.ndarray:
            return self._array

        @property
        def capsule(self) -> Any:  # expected to be py::capsule from pybind11
            return self._capsule

    # Create a wrapper to hold the array and capsule
    wrapper = ArrayCapsuleWrapper(array, obj._capsule)

    return wrapper


@pytest.fixture()
def a__cuda_array_interface__() -> Dict:
    return {"shape": (5,), "typestr": "<f4", "version": 3, "data": (None, False)}


def test_saxpy_kernel() -> None:
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


@pytest.mark.parametrize("is_int, is_float", [(True, False), (False, True)])
def test_sendreceive(is_int: bool, is_float: bool) -> None:
    if is_int:
        cupy_test_array = cp.array([41, 62, 98], dtype=cp.int32)
    elif is_float:
        cupy_test_array = cp.array([41.0, 62.0, 98.0], dtype=cp.float32)
    else:
        raise ValueError

    numpy_test_array = cp.asnumpy(cupy_test_array).copy()

    if is_int:
        returned_cupy_array = pycai.receive_and_return_cuda_array_interface_int(cupy_test_array)
    elif is_float:
        returned_cupy_array = pycai.receive_and_return_cuda_array_interface_float(cupy_test_array)
    else:
        raise ValueError

    assert hasattr(returned_cupy_array, "__cuda_array_interface__")

    np.testing.assert_allclose(numpy_test_array, cp.asnumpy(returned_cupy_array))


def test_array_without_interface() -> None:
    test_array = np.array([41.0, 62.0, 98.0], dtype=cp.float32)

    with pytest.raises(InterfaceNotImplementedError):
        pycai.receive_and_return_cuda_array_interface_float(test_array)


@pytest.mark.parametrize(
    "missing_shape, missing_typestr, missing_version, missing_data",
    [
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    ],
)
def test_missing_fields(
    a__cuda_array_interface__: Dict,
    missing_shape: bool,
    missing_typestr: bool,
    missing_version: bool,
    missing_data: bool,
) -> None:
    if missing_shape:
        del a__cuda_array_interface__["shape"]
    if missing_typestr:
        del a__cuda_array_interface__["typestr"]
    if missing_version:
        del a__cuda_array_interface__["version"]
    if missing_data:
        del a__cuda_array_interface__["data"]

    object_with_cai_attribute = SimpleNamespace(__cuda_array_interface__=a__cuda_array_interface__)

    with pytest.raises(IncompleteInterfaceError):
        pycai.receive_and_return_cuda_array_interface_float(object_with_cai_attribute)


def test_None_ptr(a__cuda_array_interface__: Dict) -> None:
    object_with_cai_attribute = SimpleNamespace(__cuda_array_interface__=a__cuda_array_interface__)

    with pytest.raises(UnRegCudaTypeError):
        pycai.receive_and_return_cuda_array_interface_float(object_with_cai_attribute)


def test_invalid_version(a__cuda_array_interface__: Dict) -> None:
    a__cuda_array_interface__["version"] = 4
    object_with_cai_attribute = SimpleNamespace(__cuda_array_interface__=a__cuda_array_interface__)

    with pytest.raises(InvalidVersionError):
        pycai.receive_and_return_cuda_array_interface_float(object_with_cai_attribute)


@pytest.mark.parametrize("shape", [(()), ((0,)), ((-2,)), ((1.5,))])
def test_invalid_shape(a__cuda_array_interface__: Dict, shape: Tuple) -> None:
    a__cuda_array_interface__["shape"] = shape
    object_with_cai_attribute = SimpleNamespace(__cuda_array_interface__=a__cuda_array_interface__)

    with pytest.raises(InvalidShapeError):
        pycai.receive_and_return_cuda_array_interface_float(object_with_cai_attribute)


@pytest.mark.parametrize("typestr", [("<R4"), ("[f4"), ("<f0"), ("qw")])
def test_invalid_typestr(a__cuda_array_interface__: Dict, typestr: str) -> None:
    a__cuda_array_interface__["typestr"] = typestr
    object_with_cai_attribute = SimpleNamespace(__cuda_array_interface__=a__cuda_array_interface__)

    with pytest.raises(InvalidTypestrError):
        pycai.receive_and_return_cuda_array_interface_float(object_with_cai_attribute)


def test_cai_on_return() -> None:
    cupy_test_array = cp.array([41.0, 62.0, 98.0], dtype=cp.float32)
    cai_before_sending = cupy_test_array.__cuda_array_interface__.copy()

    returned_cupy_array = pycai.receive_and_return_cuda_array_interface_float(cupy_test_array)

    assert cai_before_sending == returned_cupy_array.__cuda_array_interface__


def test_mismatching_dtype() -> None:
    cupy_test_array = cp.array([41.0, 62.0, 98.0], dtype=cp.float32)

    with pytest.raises(DtypeMismatchError):
        pycai.receive_and_return_cuda_array_interface_int(cupy_test_array)


def test_return_cuda_array_interface_default() -> None:
    arr = pycai.return_cuda_array_interface()
    assert arr.__cuda_array_interface__["shape"] == (3, 4, 5)
    assert arr.__cuda_array_interface__["version"] == 3
    assert not arr.__cuda_array_interface__["data"][1]  # readonly should be False


def test_return_cuda_array_interface_custom_shape() -> None:
    shape = (2, 3, 2)
    arr = pycai.return_cuda_array_interface(shape=shape)
    assert arr.__cuda_array_interface__["shape"] == shape


def test_return_cuda_array_interface_readonly() -> None:
    arr = pycai.return_cuda_array_interface(readonly=True)
    assert arr.__cuda_array_interface__["data"][1]


def test_return_cuda_array_interface_with_memory_copy_default() -> None:
    arr = pycai.return_cuda_array_interface_with_memory_copy()
    assert arr.__cuda_array_interface__["shape"] == (5,)
    assert arr.__cuda_array_interface__["version"] == 3
    assert not arr.__cuda_array_interface__["data"][1]  # readonly should be False


def test_return_cuda_array_interface_with_memory_copy_custom_vec() -> None:
    vec = [1.0, 2.0, 3.0]
    arr = pycai.return_cuda_array_interface_with_memory_copy(vec=vec)
    assert arr.__cuda_array_interface__["shape"] == (3,)


def test_return_cuda_array_interface_with_memory_copy_readonly() -> None:
    with pytest.raises(ReadOnlyAccessError):
        pycai.return_cuda_array_interface_with_memory_copy(readonly=True)


def test_memory_transfer_to_and_from_gpu() -> None:
    # Send data to GPU and retrieve it back to CPU
    vec = [1.0, 2.0, 3.0, 4.0, 5.0]
    cai = pycai.return_cuda_array_interface_with_memory_copy(vec=vec)

    arrwc = cupy_asarray_with_capsule(cai)
    del cai

    np.testing.assert_allclose(cp.asnumpy(arrwc.array), vec)

    assert arrwc.array.__cuda_array_interface__["shape"] == (5,)


def test_stress_test_large_data() -> None:
    vec = [float(i) for i in range(1000000)]
    arr = pycai.return_cuda_array_interface_with_memory_copy(vec=vec)
    assert arr.__cuda_array_interface__["shape"] == (1000000,)


def test_ownership_transfer_to_python() -> None:
    arr1 = pycai.return_cuda_array_interface()
    arr2 = pycai.return_cuda_array_interface()

    assert id(arr1) != id(arr2)
    assert hasattr(arr2, "_capsule")


def test_repeated_ownership_transfer() -> None:
    arr = pycai.return_cuda_array_interface()
    for _ in range(10):
        arr = pycai.return_cuda_array_interface()
    assert hasattr(arr, "_capsule")


def test_multiple_objects_ownership() -> None:
    objects = [pycai.return_cuda_array_interface() for _ in range(5)]
    for obj in objects:
        assert hasattr(obj, "_capsule")


def test_ensure_garbage_collection() -> None:
    arr = pycai.return_cuda_array_interface()
    capsule = arr._capsule
    del arr
    assert capsule is not None


def test_object_modification() -> None:
    arr = pycai.return_cuda_array_interface()
    arr.__cuda_array_interface__["shape"] = (6, 7, 8)
    assert arr.__cuda_array_interface__["shape"] == (6, 7, 8)


def test_interoperability_with_cupy() -> None:
    original_array = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
    returned_array = pycai.receive_and_return_cuda_array_interface_float(original_array)
    assert cp.all(original_array == returned_array)


def test_memory_consistency_with_cupy() -> None:
    original_array = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
    original_array[0, 0] = 10.0
    returned_array = pycai.receive_and_return_cuda_array_interface_float(original_array)
    assert cp.all(original_array == returned_array)


def test_shape_dtype_consistency_with_cupy() -> None:
    original_array = cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp.float32)
    returned_array = pycai.receive_and_return_cuda_array_interface_float(original_array)
    assert original_array.shape == returned_array.shape
    assert original_array.dtype == returned_array.dtype


def test_ownership_with_capsule() -> None:
    returned_cai = pycai.return_cuda_array_interface_with_memory_copy()
    assert hasattr(returned_cai, "_capsule")
    cupy_array_with_capsule = cupy_asarray_with_capsule(returned_cai)
    del returned_cai
    assert hasattr(cupy_array_with_capsule, "_capsule")


def test_cupy_asarray_with_capsule_valid_input() -> None:
    obj = pycai.return_cuda_array_interface()
    result = cupy_asarray_with_capsule(obj)

    assert isinstance(result.array, cp.ndarray)
    assert result.capsule == obj._capsule


def test_cupy_asarray_with_capsule_missing_attributes() -> None:
    mock_objs = [{}, {"__cuda_array_interface__": {}}, {"_capsule": "dummy_capsule"}]

    for mock_obj in mock_objs:
        with pytest.raises(
            ValueError, match="Input object must have both __cuda_array_interface__ and _capsule attributes."
        ):
            cupy_asarray_with_capsule(mock_obj)


def test_cupy_asarray_with_capsule_attributes() -> None:
    obj = pycai.return_cuda_array_interface()
    result = cupy_asarray_with_capsule(obj)

    assert result.array is not None
    assert result.array.shape == obj.__cuda_array_interface__["shape"]
    assert result.capsule == obj._capsule
