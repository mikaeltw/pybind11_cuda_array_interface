/***************************************************************************
* Copyright (c) 2021, Mikael Twengström                                    *
*                                                                          *
* Distributed under the terms of the XXX License.                          *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/


#pragma once

#include "pybind11/pybind11.h"

namespace pybind11 {

    class cuda_array : public object {
        public:
            PYBIND11_OBJECT_DEFAULT(cuda_array, object, PyObject_CheckBuffer)

        };

}



