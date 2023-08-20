# Pybind11 - CUDA Array Interface

## Overview

[`pybind11_cuda_array_interface`](https://github.com/mikaeltw/pybind11_cuda_array_interface) is a plugin for pybind11 that facilitates effortless exchange of arrays between Python and C++ environments. It specifically targets objects or arrays that implement the [`__cuda_array_interface__`](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html), ensuring smooth interaction with GPU data structures.

## Features

- **Automatic Conversion**: No need for manual data type conversions. The plugin handles it seamlessly.
- **Lifecycle management**: Specifically designed to work with the __cuda_array_interface__, by defining a `cuda_array_t<T>` type for the C++ side managing the reference count intrinsically.
- **Header-only**: Just include the header and you're set. No heavy installations or configurations required.

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

```python

```



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

## License
This project is licensed under the BSD-3-Clause License. See the [LICENSE](LICENSE) file for details.