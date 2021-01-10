/***************************************************************************
* Copyright (c) 2021, Mikael Twengström                                    *
*                                                                          *
* Distributed under the terms of the XXX License.                          *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/


#pragma once

#include "pybind11/pybind11.h"
#include <cuda_runtime.h>

inline const char *enum_to_string(const cudaError_t &error)
{
    return cudaGetErrorName(error);
}

template<typename T>
inline void die_on_unsuccessful_enum(T result, const char *func, const char *file, int line)
{
    if (result) {
        fprintf(stderr, "CUDA driver error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), enum_to_string(result), func);
        exit(EXIT_FAILURE);
    }
}

#define cudaCheckErrors(val) die_on_unsuccessful_enum(val, #val, __FILE__, __LINE__)

class PythonException : public std::exception {
    public:
        explicit PythonException(const char *message, PyObject *PyExc) : m_message{message}, m_PyExc{PyExc} {}
        const char *what() const noexcept override {return m_message.c_str();}
        PyObject *get_error_type() const noexcept {return m_PyExc;}
    private:
        std::string m_message{};
        PyObject *m_PyExc{};
};

void register_and_throw_exception(const char *error_message)
{
    pybind11::register_exception_translator([](std::exception_ptr exc_ptr) {
        try {
            if (exc_ptr) std::rethrow_exception(exc_ptr);
        } catch (const PythonException &exc) {
            PyErr_SetString(PyExc_NotImplementedError, exc.what());
        }
    });
    throw PythonException(error_message, PyExc_NotImplementedError);
}

int PyObject_CheckCudaArray(PyObject *obj)
{
    return PyObject_HasAttrString(obj, "__cuda_array_interface__");
}

namespace pybind11 {
    struct cuda_array_info {
        void *device_ptr = nullptr;   // Pointer to the underlying storage on device
        ssize_t itemsize = 0;         // Size of individual items in bytes
        ssize_t size = 0;             // Total number of entries
        ssize_t version = 0;          // Version of the cuda_array_interface
        std::string format;           // For homogeneous buffers, this should be set to format_descriptor<T>::format()
        ssize_t ndim = 0;             // Number of dimensions
        std::vector<ssize_t> shape;   // Shape of the tensor (1 entry per dimension)
        std::vector<ssize_t> strides; // Number of bytes between adjacent entries (for each per dimension)
        bool readonly = false;        // flag to indicate if the underlying storage may be written to

        cuda_array_info() = default;

        cuda_array_info(pybind11::dict cuda_dict) {
            extract_dict_info(cuda_dict);
        }

        private:
            void parse_typestr(pybind11::str typestr)
            {
                auto typestr_cpp = typestr.cast<std::string>();
                std::string byteorder = typestr_cpp.substr(0,1);
                format = typestr_cpp.substr(1,1);
                itemsize = std::stoi(typestr_cpp.substr(2, std::string::npos));
            }

            void extract_device_ptr(pybind11::tuple data)
            {
                readonly = data[1].cast<bool>();
                device_ptr = reinterpret_cast<void *>(data[0].cast<int64_t>());
                cudaPointerAttributes attributes;
                cudaCheckErrors(cudaPointerGetAttributes(&attributes, const_cast<const void *>(device_ptr)));

                if (attributes.devicePointer == nullptr || attributes.devicePointer != device_ptr) {
                    throw std::runtime_error("Illegal device ptr retrieved!");
                }
            }

            void extract_dict_info(pybind11::dict cuda_dict)
            {
                version = cuda_dict["version"].cast<ssize_t>();

                pybind11::tuple py_shape = cuda_dict["shape"];
                ssize_t dim_size = 1;
                for (auto item : py_shape) {
                    dim_size *= item.cast<ssize_t>();
                    shape.push_back(dim_size);
                    ++ndim;
                }
                size = dim_size;

                if (cuda_dict["strides"].is_none()) {
                    strides.push_back(0);
                } else {
                    pybind11::tuple py_strides = cuda_dict["strides"];
                    for (auto item : py_strides) {
                        strides.push_back(item.cast<ssize_t>());
                    }
                }

                if (cuda_dict.contains("mask")) {
                    register_and_throw_exception("Attribute 'mask' in __cuda_array_interface__ is currently not supported!");
                }

                pybind11::str typestr = cuda_dict["typestr"];
                parse_typestr(typestr);

                if (std::string(pybind11::str(pybind11::tuple(pybind11::list(cuda_dict["descr"])[0])[1])).compare(std::string(typestr)) != 0) {
                    register_and_throw_exception("Non trivial types of descr field is not supported");
                }

                pybind11::tuple data = cuda_dict["data"];
                extract_device_ptr(data);
            }
    };

    class cuda_array : public object {
        public:
            PYBIND11_OBJECT_DEFAULT(cuda_array, object, PyObject_CheckCudaArray)

            cuda_array_info get_cuda_array_interface()
            {
                auto obj_ptr = this->m_ptr;
                auto cuda_array_interface_info = PyObject_GetAttrString(obj_ptr, "__cuda_array_interface__");
                pybind11::dict cuda_dict = reinterpret_borrow<pybind11::dict>(cuda_array_interface_info);
                return cuda_array_info(cuda_dict);
            }
    };
}
