/***************************************************************************
* Copyright (c) 2021, Mikael Twengström                                    *
*                                                                          *
* Distributed under the terms of the XXX License.                          *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/


#pragma once

#define PY_SSIZE_T_CLEAN
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <functional>

#include <cuda.h>

namespace py = pybind11;

namespace cuerrutil {

    inline const char *enum_to_string(const CUresult& error)
    {
        const char *ret = nullptr;
        cuGetErrorName(error, &ret);
        return ret;
    }

    template <typename T>
    inline void throw_on_unsuccessful_enum(T result, const char *func, const char *file, int line)
    {
        if (result) {
            std::stringstream ss_error;
            ss_error << "CUDA error at " << file << ":" << line
                    << " code=" << static_cast<unsigned int>(result) << "(" << enum_to_string(result)
                    << ") \"" << func << "\"";

            throw std::runtime_error(ss_error.str());
        }
    }
}

#define checkCudaErrors(val) cuerrutil::throw_on_unsuccessful_enum(val, #val, __FILE__, __LINE__)

namespace cai {

    //Forward declarations
    struct cuda_array_t;
    //template <> struct py::detail::type_caster<cuda_array_t>;

    struct cuda_memory_handle {
        private:
            CUdeviceptr ptr;
            std::function<void(CUdeviceptr)> deleter;

            // Constructor for C++-created objects (default deleter)
            cuda_memory_handle(CUdeviceptr ptr)
                : ptr(ptr), deleter([](CUdeviceptr ptr) { cuMemFree(ptr); }) {}

            // Constructor for Python-created objects (explicit do-nothing deleter)
            cuda_memory_handle(CUdeviceptr ptr, std::function<void(CUdeviceptr)> deleter)
                : ptr(ptr), deleter(deleter) {}

            friend struct cuda_array_t;
            friend struct py::detail::type_caster<cuda_array_t>;

        public:
            ~cuda_memory_handle() {
                deleter(ptr);
            }

        protected:
            // Factory method
            template<typename... Args>
            static std::shared_ptr<cuda_memory_handle> make_shared_handle(Args&&... args) {
                return std::shared_ptr<cuda_memory_handle>(new cuda_memory_handle(std::forward<Args>(args)...));
            }
    };

    struct cuda_array_t {
        private:
            std::shared_ptr<cuda_memory_handle> handle;
            std::vector<size_t> shape;
            std::string typestr;
            bool readonly;
            int version;
            py::object py_obj;

            CUdeviceptr ptr() const {
                return handle->ptr;
            }

            cuda_array_t() {};

            friend struct py::detail::type_caster<cuda_array_t>;

        public:
            const std::vector<size_t>& get_shape() {
                return shape;
            }

            const py::dtype get_dtype() {
                return py::dtype(typestr);
            }

            bool is_readonly() {
                return readonly;
            }

            int get_version() {
                return version;
            }

            size_t size_of_shape() {
                return std::accumulate(shape.begin(), shape.end(), static_cast<std::size_t>(1), std::multiplies<>());
            }

            template <typename T>
            T* get_compatible_typed_pointer() {
                py::dtype expected_dtype = py::dtype::of<T>();
                py::dtype dt(this->typestr);

                if (!expected_dtype.is(dt)) {
                    std::stringstream error_ss;
                    error_ss << "Mismatching dtypes. " << "Expected the dtype: "
                            << py::str(expected_dtype).cast<std::string>() << " corresponding"
                            << " to a C++ " << typeid(T).name() << " which is not compatible "
                            << "with the supplied dtype " << py::str(dt).cast<std::string>() << "\n";
                    throw std::runtime_error(error_ss.str());
                }

                return reinterpret_cast<T*>(this->ptr());
            }
    };

}

template <> struct py::detail::type_caster<cai::cuda_array_t> {
public:
    PYBIND11_TYPE_CASTER(cai::cuda_array_t, _("cai::cuda_array_t"));

    // Python -> C++ conversion
    bool load(py::handle src, bool) {

        py::object obj = py::reinterpret_borrow<py::object>(src);

        if (!py::hasattr(obj, "__cuda_array_interface__")) {
            throw std::runtime_error("Provided Python Object does not have a __cuda_array_interface__");
        }

        py::object interface = obj.attr("__cuda_array_interface__");
        py::dict iface_dict = interface.cast<py::dict>();

        if (!iface_dict.contains("data") || !iface_dict.contains("shape") ||
            !iface_dict.contains("typestr") || !iface_dict.contains("version")) {
            throw std::runtime_error("At least one of the mandatory fields: data, shape, typestr or version is missing from from the provided __cuda_array_interface__");
        }

        //Extract the version key from the cuda array dict
        value.version = iface_dict["version"].cast<int>();
        if (value.version != 3) {
            throw std::runtime_error("Unsupported __cuda_array_interface__ version != 3");
        }

        //Extract the shape key from the cuda array dict
        for (py::handle h : iface_dict["shape"].cast<py::tuple>()) {
            value.shape.emplace_back(h.cast<size_t>());
        }

        //Extract the typestr key from the cuda array dict
        value.typestr = iface_dict["typestr"].cast<std::string>();

        //Extract the data key from the cuda array dict
        py::tuple data = iface_dict["data"].cast<py::tuple>();
        CUdeviceptr inputPtr = data[0].cast<CUdeviceptr>();
        CUdeviceptr devicePtr;
        checkCudaErrors(cuPointerGetAttribute(&devicePtr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, inputPtr));

        value.handle = cai::cuda_memory_handle::make_shared_handle(devicePtr, [](CUdeviceptr){});

        value.readonly = data[1].cast<bool>();

        // Keep a reference to the original Python object to prevent it from being garbage collected
        value.py_obj = obj;

        return true;
    }

    // C++ -> Python conversion
    static py::handle cast(const cai::cuda_array_t& src, return_value_policy /* policy */, handle /* parent */) {
        CUcontext ctx;
        checkCudaErrors(cuCtxGetCurrent(&ctx));
        if (!ctx) {
            throw std::runtime_error("Improper current CUDA context");
        }

        // If py_obj of src is set it means it originates from Python and the object can thus be released.
        if (!src.py_obj.is_none()) {
            return src.py_obj;
        }

        CUdeviceptr deviceptr;
        checkCudaErrors(cuPointerGetAttribute(&deviceptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, src.ptr()));

        py::dict interface;

        interface["shape"] = py::tuple(py::cast(src.shape));
        interface["typestr"] = py::str(py::cast(src.typestr));
        interface["data"] = py::make_tuple(deviceptr, src.readonly);
        interface["version"] = 3;

        // Assuming src was created in C++, src.handle owns the CUDA memory and should
        // be wrapped in a py::capsule to transfer ownership to Python.
        py::capsule caps(src.handle.get(), [](void* p) {
            delete reinterpret_cast<cai::cuda_memory_handle*>(p);
        });

        // Null out the cuda_memory_handle pointer in the shared_ptr to prevent it from being freed when src is destroyed.
        const_cast<std::shared_ptr<cai::cuda_memory_handle>&>(src.handle).reset();

        // Create an instance of a Python object that can hold arbitrary attributes
        py::object types = py::module::import("types");
        py::object caio = types.attr("SimpleNamespace")();
        caio.attr("__cuda_array_interface__") = interface;

        // Make the SimpleNamespace object owns the capsule to prevent it from being garbage collected
        caio.attr("_capsule") = caps;

        return caio.release();
    }
};
