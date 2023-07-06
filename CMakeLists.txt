##########################
### Define the project ###
##########################

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(pybind11_cuda_array_interface LANGUAGES CUDA CXX)

set(PYBIND11_CUDA_ARRAY_INTERFACE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(${PROJECT_NAME}_VERSION_MAJOR 0)
set(${PROJECT_NAME}_VERSION_MINOR 0)
set(${PROJECT_NAME}_VERSION_PATCH 1)
set(${PROJECT_NAME}_VERSION ${${PROJECT_NAME}_VERSION_MAJOR}.${${PROJECT_NAME}_VERSION_MINOR}.${${PROJECT_NAME}_VERSION_PATCH})

set(CMAKE_VERBOSE_MAKEFILE ON)

#######################################
### Get operating system properties ###
#######################################
set(GET_OS_INFO_CMD lsb_release -a)
execute_process(COMMAND ${GET_OS_INFO_CMD} OUTPUT_VARIABLE OS_PROPERTIES OUTPUT_STRIP_TRAILING_WHITESPACE)
message("========= DETECTED OS =========")
message("${OS_PROPERTIES}")
message("===============================")

######################################
### Enable and define CUDA details ###
######################################

set(CMAKE_CXX_VERSION 14)
set(CMAKE_CXX_STANDARD ${CMAKE_CXX_VERSION})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_VERSION})
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
else()
    message(FATAL_ERROR "No CUDA compiler found")
endif()

####################
### Dependencies ###
####################

set(pybind11_REQUIRED_VERSION 2.9.2)
set(python_REQUIRED_VERSION 3.8)
set(cuda_REQUIRED_VERSION 10.1)

if (NOT TARGET pybind11::pybind11)
    find_package(pybind11 ${pybind11_REQUIRED_VERSION} REQUIRED)
endif()

if(NOT TARGET Python3::Interpreter OR NOT TARGET Python3::Python)
    set(Python3_FIND_VIRTUALENV ONLY)
    find_package(Python3 ${python_REQUIRED_VERSION} REQUIRED COMPONENTS Interpreter Development)
endif()

if(NOT TARGET CUDA::toolkit)
    find_package(CUDA ${cuda_REQUIRED_VERSION} REQUIRED)
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
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "61")

##############################
### Define and build Tests - GoogleTests - Bindings for PyTest ###
##############################

OPTION(BUILD_GTESTS "Build pybind11_cuda_array_interface googletests" OFF)
OPTION(DOWNLOAD_GTEST "Download googletests" OFF)
OPTION(BUILD_PYTESTS "Build pybind11_cuda_array_interface bindings for pytest" OFF)

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    set(BUILD_GTESTS ON)
endif()

if(BUILD_GTESTS OR BUILD_PYTESTS)
    add_subdirectory(tests)
endif()

#################################
# Installation of the library ###
#################################

include(CMakePackageConfigHelpers)

set(PYBIND11_CUDA_ARRAY_INTERFACE_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}" CACHE STRING "install path for pybind11_cuda_array_interfaceConfig.cmake")

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}-targets PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

export(EXPORT ${PROJECT_NAME}-targets FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

# Configure 'pybind11_cuda_array_interfaceConfig.cmake' for a build tree
set(PYBIND11_CUDA_ARRAY_INTERFACE_CONFIG_CODE "####### Expanded from \@PYBIND11_CUDA_ARRAY_INTERFACE_CONFIG_CODE\@ #######\n")
set(PYBIND11_CUDA_ARRAY_INTERFACE_CONFIG_CODE "${PYBIND11_CUDA_ARRAY_INTERFACE_CONFIG_CODE}set(CMAKE_MODULE_PATH \"${CMAKE_CURRENT_SOURCE_DIR}/cmake;\${CMAKE_MODULE_PATH}\")\n")
set(PYBIND11_CUDA_ARRAY_INTERFACE_CONFIG_CODE "${PYBIND11_CUDA_ARRAY_INTERFACE_CONFIG_CODE}##################################################")
configure_package_config_file(${PROJECT_NAME}Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" INSTALL_DESTINATION ${PROJECT_BINARY_DIR})

# Configure 'pybind11_cuda_array_interfaceConfig.cmake' for an install tree
set(PYBIND11_CUDA_ARRAY_INTERFACE_CONFIG_CODE "")
configure_package_config_file(${PROJECT_NAME}Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}Config.cmake" INSTALL_DESTINATION ${PYBIND11_CUDA_ARRAY_INTERFACE_CMAKECONFIG_INSTALL_DIR})

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake VERSION ${${PROJECT_NAME}_VERSION} COMPATIBILITY AnyNewerVersion ARCH_INDEPENDENT)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${PROJECT_NAME}Config.cmake ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake DESTINATION ${PYBIND11_CUDA_ARRAY_INTERFACE_CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}-targets FILE ${PROJECT_NAME}Targets.cmake DESTINATION ${PYBIND11_CUDA_ARRAY_INTERFACE_CMAKECONFIG_INSTALL_DIR})
