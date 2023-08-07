#[[
## Copyright (c) 2021, Mikael Twengström
## All rights reserved.
## This file is part of pybind11_cuda_array_interface and is distributed under the
## BSD-3 Clause License. For full terms see the included LICENSE file.
#]]

cmake_minimum_required(VERSION ${CMAKE_MVERSION} FATAL_ERROR)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

include(CheckCXXCompilerFlag)
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_${CMAKE_CXX_STANDARD})

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
    CMAKE_CXX_COMPILER_ID MATCHES "Intel")

    target_compile_options(project_options INTERFACE -Wunused-parameter -Wextra -Wreorder -Wconversion -Wsign-conversion)
    CHECK_CXX_COMPILER_FLAG(-march=native HAS_MARCH_NATIVE)

    if (HAS_MARCH_NATIVE)
        target_compile_options(project_options INTERFACE -march=native)
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(project_options INTERFACE /EHsc /MP /bigobj)
    target_link_options(project_options INTERFACE /MANIFEST:NO)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(pybind11_cuda_array_interface-test)
    enable_testing()
    find_package(pybind11_cuda_array_interface REQUIRED CONFIG)
endif()

if(BUILD_GTESTS)
    configure_file(gtest/getGTest.cmake.in gtest-get/CMakeLists.txt)

    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gtest-get
    )
    if(result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()

    execute_process(COMMAND ${CMAKE_COMMAND} --build .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gtest-get
    )
    if(result)
        message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/gtest-src
                    ${CMAKE_CURRENT_BINARY_DIR}/gtest-build EXCLUDE_FROM_ALL
    )

    set(GTEST_INCLUDE_DIRS "${gtest_SOURCE_DIR}/include")
    set(GTEST_BOTH_LIBRARIES gtest_main gtest)
    set(EXECNAME run_gtest_cai)
    set(PYBIND11_CUDA_ARRAY_INTERFACE_GTESTS sources/gtest_pybind11_cuda_array_interface.cpp)

    add_executable(${EXECNAME} ${PYBIND11_CUDA_ARRAY_INTERFACE_GTESTS})
    add_dependencies(${EXECNAME} gtest_main)
    target_link_libraries(${EXECNAME} PRIVATE
                project_options
                Python3::Python
                ${GTEST_BOTH_LIBRARIES}
                pybind11::pybind11
                pybind11::embed
                CUDA::cudart
    )
    target_include_directories(${EXECNAME} PRIVATE
                               ${PYBIND11_CUDA_ARRAY_INTERFACE_INCLUDE_DIR}
                               ${GTEST_INCLUDE_DIRS}
    )
    add_custom_command(TARGET ${EXECNAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                               $<TARGET_FILE:${EXECNAME}>
                               ${CMAKE_SOURCE_DIR}/tests/gtest/$<TARGET_FILE_NAME:${EXECNAME}>
    )

endif()

if(BUILD_PYTESTS)
    set(MODULENAME pycai)
    set(CPPSOURCES sources/pytest_pybind11_cuda_array_interface.cpp)
    set(CUSOURCES sources/test_kernels.cu)
    set(LIBS Python3::Python pybind11::pybind11 pybind11::embed CUDA::cudart)

    # TODO: The following module from pybind11 does not respect the CMAKE_CUDA_ARCHITECTURES
    # variable which handles which archs to build and link to. This results in missing linking
    # of device code in the produced module. Use add_library until bug resolved.
    # pybind11_add_module(${MODULENAME} ${CPPSOURCES} ${CUSOURCES})
    # The problem lies in the linking of .cu sources using pybind11::lto and pybind11::thin_lto,
    # since lto is not fully supported by nvcc as far as I know.

    add_library(${MODULENAME} MODULE ${CPPSOURCES} ${CUSOURCES})
    target_link_libraries(${MODULENAME} PRIVATE project_options ${LIBS} )
    target_include_directories(${MODULENAME} PRIVATE
                               ${PYBIND11_CUDA_ARRAY_INTERFACE_INCLUDE_DIR}
    )
    set_target_properties(${MODULENAME} PROPERTIES PREFIX "")
    add_custom_command(TARGET ${MODULENAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                               $<TARGET_FILE:${MODULENAME}>
                               ${CMAKE_SOURCE_DIR}/tests/pytest/$<TARGET_FILE_NAME:${MODULENAME}>
    )

endif()
