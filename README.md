# Pybind11 - CUDA Array Interface

[![CI](https://github.com/mikaeltw/pybind11_cuda_array_interface/actions/workflows/ci.yaml/badge.svg)](https://github.com/mikaeltw/pybind11_cuda_array_interface/actions/workflows/ci.yaml)
[![Linting](https://github.com/mikaeltw/pybind11_cuda_array_interface/actions/workflows/linting.yaml/badge.svg)](https://github.com/mikaeltw/pybind11_cuda_array_interface/actions/workflows/linting.yaml)
[![PyPI](https://img.shields.io/pypi/v/pybind11-cuda-array-interface.svg?logo=PyPi)](https://pypi.org/project/pybind11-cuda-array-interface/)
[![Python](https://img.shields.io/pypi/pyversions/pybind11-cuda-array-interface?logo=Python)](https://pypi.org/project/pybind11-cuda-array-interface/)
[![License](https://img.shields.io/pypi/l/pybind11-cuda-array-interface.svg)](https://pypi.org/project/pybind11-cuda-array-interface/)

## Overview

[`pybind11_cuda_array_interface`](https://github.com/mikaeltw/pybind11_cuda_array_interface) is a plugin for pybind11 that facilitates effortless exchange of arrays between Python and C++ environments. It specifically targets objects or arrays that implement the [`__cuda_array_interface__`](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html), ensuring smooth interaction with GPU data structures.

## Features

- **Automatic Conversion**:

    No need for manual data type conversions. The plugin handles it seamlessly.
- **Lifecycle management**:

    Specifically designed to work with the `__cuda_array_interface__`, by defining a `cuda_array_t<T>` type for the C++ side managing the reference count intrinsically.
- **Header-only**:

    Just include the header and you're set. No heavy installations or configurations required.

## Usage and making bindings

```cpp
#include "pybind11_cuda_array_interface/pybind11_cuda_array_interface.hpp"

namespace py = pybind11;

template <typename T>
cai::cuda_array_t<T> receive_and_return_cuda_array_interface(cai::cuda_array_t<T> s_cai)
{
    return s_cai;
}

template <typename T>
cai::cuda_array_t<T> return_cuda_array_interface(std::vector<size_t> shape = {3, 4, 5},
                                                 bool readonly = false, int version = 3)
{
    cai::cuda_array_t<T> s_cai(shape, readonly, version);
    return s_cai;
}

```

**Bindings:**

```cpp
PYBIND11_MODULE(pycai, module)
{
    module.doc() = "Module for __cuda_array_interface__";
    caiexcp::register_custom_cuda_array_interface_exceptions(module);

    module.def("receive_and_return_cuda_array_interface_float",
               &receive_and_return_cuda_array_interface<float>,
               "Function accepts a cuda_array_t and immediately just sends it back",
               py::arg("s"));

    module.def(
        "return_cuda_array_interface",
        &return_cuda_array_interface<float>,
        "Constructs a cuda_array_t in C++ and returns it to Python",
        py::arg_v("shape", std::vector<size_t>({3, 4, 5}), "std::vector<size_t>({3, 4, 5})"),
        py::arg("readonly") = false,
        py::arg("version") = 3);
}
```

From python you can now use these functions to send and receive arrays implementing the [`__cuda_array_interface__`](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html) defined by Numba.

## Features and the `cuda_array_t<T>` class

**Custom exceptions:**

The plugin defines a number of internal exceptions and a utility function is provided for exporting them to Python via `caiexcp::register_custom_cuda_array_interface_exceptions(module);`. For a complete list please refer to the source code [here](include/pybind11_cuda_array_interface/pybind11_cuda_array_interface.hpp).

**cuda_array_t\<T\>:**

The following public utilities are provided by `cuda_array_t<T>`:
```cpp
cuda_array_t(std::vector<size_t> shape, const bool readonly = false, const int version = 3)
        : shape(std::move(shape)), readonly(readonly), version(version)
```
, where `shape`, `readonly` and `version` correspond to their counterparts of the `__cuda_array_interface__`.

```cpp
const std::vector<size_t> &get_shape() const
```
, get method to return the shape of the array stored as an `std::vector`.
```cpp
py::dtype get_dtype() const
```
, get method to return the `dtype` corresponding to the provided type `T`.
```cpp
bool is_readonly() const
```
, get method to  return the boolean state of the `readonly` variable of `__cuda_array_interface__`.
```cpp
int get_version() const
```
, get method to return the version of `__cuda_array_interface__`.
```cpp
size_t size_of_shape() const
```
, get method to return the total number of elements in the pointed to cuda_array.
```cpp
T *get_compatible_typed_pointer()
const T *get_compatible_typed_pointer() const
```
, get method to return a raw pointer of type `T` as `const` or `non-const` depending on the value of `readonly`.

**Return type:**

The `__cuda_array_interface__` is represented by the `cuda_array_t<T>` class in C++, but the integrity of the Python array implementing the `__cuda_array_interface__` is respected when said object is returned to Python from C++. I.e. objects sent from Python to C++ and at a later stage returned to Python as a `cuda_array_t<T>`, will still return the actual Python object sent in the first place. However, should a `cuda_array_t<T>` instance be defined or originating from C++ it will be returned to Python as a `types.SimpleNamespace` implementing the `__cuda_array_interface__` and a `py::capsule` object to handle the ownership transfer.

Using a [CuPy](https://cupy.dev/) array the following wrapper class might prove useful:

```python
from typing import Any, Protocol

import cupy as cp

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
```

# Installations
Depending on your preferred choice the package can be installed in several ways given that three main dependencies are met

- [pybind11](https://github.com/pybind/pybind11) >=2.11.1
- [Python](https://github.com/python/cpython) >=3.8
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) >=10.2

## Install from PyPI
```bash
pip install pybind11-cuda-array-interface
```
which will install the header files such that [CMake](https://cmake.org/) can find them.

## Install from source
```bash
git clone git@github.com:mikaeltw/pybind11_cuda_array_interface.git
```
or
```bash
git clone https://github.com/mikaeltw/pybind11_cuda_array_interface.git
```
Then
```bash
python -m pip install .
```
or (Provided that you have CMake >=3.18 installed)
```bash
mkdir build
cd build
cmake ..
make install
```

## Install by copying headers
You can also copy the headers manually from `include/pybind11_cuda_array_interface/*`.

## Linting and tests
Please note that the tests require a functional NVIDIA GPU of at least the Pascal generation.

**Install dependencies and compile tests:**
```bash
python tests/install_cupy.py
python linting/install_linters.py
BUILD_GTESTS=ON BUILD_PYTESTS=ON python -m pip install .[test]
```

**Run tests:**
```bash
python -m pytest
```
and
```bash
tests/gtest/run_gtest_cai
```

**Run Python linting:**
```bash
python linting/check_python_linting.py
```

**Clang-tidy & Clang-format:**

No utility command provided but implemented in [.github/workflows/linting.yaml](.github/workflows/linting.yaml).


## Contributing
Issues and PRs are welcomed. If opening a PR, please do so from a fork of the repository.

## License
This project is licensed under the BSD-3-Clause License. See the [LICENSE](LICENSE) file for details.