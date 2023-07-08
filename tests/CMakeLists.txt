############################################################################
# Copyright (c) 2021, Mikael Twengström                                    #
#                                                                          #
# Distributed under the terms of the XXX.                                  #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(pybind11_cuda_array_interface-test)
    enable_testing()
    find_package(pybind11_cuda_array_interface REQUIRED CONFIG)
endif()

message(STATUS "Forcing tests build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if(CMAKE_CXX_COMPILER_ID MATCHES Clang OR CMAKE_CXX_COMPILER_ID MATCHES GNU OR CMAKE_CXX_COMPILER_ID MATCHES Intel)
    add_compile_options(-Wunused-parameter -Wextra -Wreorder -Wconversion -Wsign-conversion)

    CHECK_CXX_COMPILER_FLAG(-march=native HAS_MARCH_NATIVE)
    if (HAS_MARCH_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    add_compile_options(/EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    if(DOWNLOAD_GTEST)
        # Download and unpack googletest at configure time
        configure_file(gtest/getGTest.cmake.in gtest-get/CMakeLists.txt)
    else()
        # Copy local source of googletest at configure time
        configure_file(gtest/cpGTest.cmake.in gtest-get/CMakeLists.txt)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gtest-get )
    if(result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gtest-get )
    if(result)
        message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/gtest-src
                     ${CMAKE_CURRENT_BINARY_DIR}/gtest-build EXCLUDE_FROM_ALL)

    set(GTEST_INCLUDE_DIRS "${gtest_SOURCE_DIR}/include")
    set(GTEST_BOTH_LIBRARIES gtest_main gtest)
elseif(BUILD_GTESTS)
    find_package(GTest REQUIRED)
endif()

if(BUILD_GTESTS)
    set(EXECNAME run_gtest_cai)

    include_directories(${GTEST_INCLUDE_DIRS} SYSTEM)

    set(PYBIND11_CUDA_ARRAY_INTERFACE_GTESTS sources/gtest_pybind11_cuda_array_interface.cpp)

    add_executable(${EXECNAME} ${PYBIND11_CUDA_ARRAY_INTERFACE_GTESTS})
    if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
        add_dependencies(${EXECNAME} gtest_main)
    endif()

    target_link_libraries(${EXECNAME} Python3::Python ${GTEST_BOTH_LIBRARIES} pybind11::pybind11 pybind11::embed cuda)
    target_include_directories(${EXECNAME} PRIVATE ${PYBIND11_CUDA_ARRAY_INTERFACE_INCLUDE_DIR})

    add_custom_command(TARGET ${EXECNAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                               $<TARGET_FILE:${EXECNAME}>
                               ${CMAKE_SOURCE_DIR}/tests/gtest/$<TARGET_FILE_NAME:${EXECNAME}>)

endif()

if(BUILD_PYTESTS)
    set(MODULENAME pycai)
    set(HPPSOURCES sources/test_kernels.hpp)
    set(CPPSOURCES sources/pytest_pybind11_cuda_array_interface.cpp)
    set(CUSOURCES sources/test_kernels.cu)
    set(LIBS -lcuda)

    pybind11_add_module(${MODULENAME} ${CPPSOURCES} ${CUSOURCES} ${HPPSOURCES})

    set_target_properties(${MODULENAME} PROPERTIES CUDA_ARCHITECTURES "61")
    target_link_libraries(${MODULENAME} PRIVATE ${LIBS})
    target_include_directories(${MODULENAME} PRIVATE ${PYBIND11_CUDA_ARRAY_INTERFACE_INCLUDE_DIR})

    add_custom_command(TARGET ${MODULENAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy
                               $<TARGET_FILE:${MODULENAME}>
                               ${CMAKE_SOURCE_DIR}/tests/pytest/$<TARGET_FILE_NAME:${MODULENAME}>)

    find_program(PYTEST_EXECUTABLE NAMES pytest)

    if(NOT PYTEST_EXECUTABLE)
        message(FATAL_ERROR "pytest not found!")
    endif()
endif()
