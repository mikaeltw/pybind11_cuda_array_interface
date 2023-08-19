#[[
## Copyright (c) 2021, Mikael Twengström
## All rights reserved.
## This file is part of pybind11_cuda_array_interface and is distributed under the
## BSD-3 Clause License. For full terms see the included LICENSE file.
#]]

#######################################################
### Set default values for version for dependencies ###
#######################################################
macro(set_if_not_defined var default_value)
    if(NOT DEFINED ${var})
        set(${var} ${default_value})
    endif()
endmacro()

set_if_not_defined(CMAKE_MVERSION 3.18)
set_if_not_defined(PYBIND11_MVERSION 2.9.2)
set_if_not_defined(PYTHON_MVERSION 3.8)
set_if_not_defined(CUDA_MVERSION 10.1)
set_if_not_defined(PACKAGE_MVERSION 0.0.0-dev)
set_if_not_defined(CXX_MVERSION 17)

##########################
### Define the project ###
##########################
cmake_minimum_required(VERSION ${CMAKE_MVERSION} FATAL_ERROR)

# Set the variable to generate compile_commands.json for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(pybind11_cuda_array_interface LANGUAGES CUDA CXX)

set(PYBIND11_CUDA_ARRAY_INTERFACE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(${PROJECT_NAME}_VERSION ${PACKAGE_MVERSION})
set(CMAKE_VERBOSE_MAKEFILE ON)

#######################################
### Get operating system properties ###
#######################################
set(GET_OS_INFO_CMD lsb_release -a)
execute_process(COMMAND ${GET_OS_INFO_CMD}
                OUTPUT_VARIABLE OS_PROPERTIES
                OUTPUT_STRIP_TRAILING_WHITESPACE)
message("========= DETECTED OS =========")
message("${OS_PROPERTIES}")
message("===============================")

######################################
### Enable and define CUDA details ###
######################################

set(CMAKE_CXX_STANDARD ${CXX_MVERSION})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD} CACHE STRING "")
    set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE STRING "")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 CACHE STRING "")
else()
    message(FATAL_ERROR "No CUDA compiler found")
endif()

####################
### Dependencies ###
####################
set(Python3_FIND_VIRTUALENV FIRST)
if(NOT TARGET Python3::Interpreter OR NOT TARGET Python3::Python)
    find_package(Python3 ${PYTHON_MVERSION} REQUIRED COMPONENTS Interpreter Development)
endif()

if (NOT TARGET pybind11::pybind11)
    find_package(pybind11 ${PYBIND11_MVERSION} REQUIRED)
endif()

if(NOT TARGET CUDA::toolkit)
    find_package(CUDAToolkit ${CUDA_MVERSION} REQUIRED)
endif()

############################
### Customise the target ###
############################
include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib; ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(PYBIND11_CUDA_ARRAY_INTERFACE_HEADERS include/pybind11_cuda_array_interface/pybind11_cuda_array_interface.hpp)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME}
                           INTERFACE
                           ${PYTHON_INCLUDE_DIRS}
                           $<BUILD_INTERFACE:${PYBIND11_CUDA_ARRAY_INTERFACE_INCLUDE_DIR}>
                           $<INSTALL_INTERFACE:include>)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${PYBIND11_CUDA_ARRAY_INTERFACE_HEADERS}")

##################################################################
### Define and build Tests - GoogleTests - Bindings for PyTest ###
##################################################################
OPTION(BUILD_GTESTS "Build pybind11_cuda_array_interface googletests" OFF)
OPTION(BUILD_PYTESTS "Build pybind11_cuda_array_interface bindings for pytest" OFF)

if(BUILD_GTESTS OR BUILD_PYTESTS)
    add_subdirectory(tests)
endif()

#################################
# Installation of the library ###
#################################
include(CMakePackageConfigHelpers)
set(PYBIND11_CUDA_ARRAY_INTERFACE_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}" CACHE STRING "install path for pybind11_cuda_array_interfaceConfig.cmake")
install(TARGETS
            ${PROJECT_NAME}
        EXPORT
            ${PROJECT_NAME}-targets
        PUBLIC_HEADER
        DESTINATION
            ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

export(EXPORT
           ${PROJECT_NAME}-targets
       FILE
           "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
)

# Define and setup the build tree
set(PYBIND11_CUDA_ARRAY_INTERFACE_CONFIG_CODE "set(CMAKE_MODULE_PATH \"${CMAKE_CURRENT_SOURCE_DIR}/cmake;\${CMAKE_MODULE_PATH}\")")
configure_package_config_file(${PROJECT_NAME}Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" INSTALL_DESTINATION ${PROJECT_BINARY_DIR})

# Define and setup the install tree
set(PYBIND11_CUDA_ARRAY_INTERFACE_CONFIG_CODE "")
configure_package_config_file(${PROJECT_NAME}Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}Config.cmake" INSTALL_DESTINATION ${PYBIND11_CUDA_ARRAY_INTERFACE_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake VERSION ${${PROJECT_NAME}_VERSION} COMPATIBILITY AnyNewerVersion ARCH_INDEPENDENT)

install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}Config.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION
            ${PYBIND11_CUDA_ARRAY_INTERFACE_CMAKECONFIG_INSTALL_DIR}
)

install(EXPORT
            ${PROJECT_NAME}-targets
        FILE
            ${PROJECT_NAME}Targets.cmake
        DESTINATION
            ${PYBIND11_CUDA_ARRAY_INTERFACE_CMAKECONFIG_INSTALL_DIR}
)
