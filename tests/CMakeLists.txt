cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(pybind11_cuda_array_interface_test LANGUAGES CUDA CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake-modules)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ".")

##################################################################
### Get operating system properties                            ###
##################################################################
set(GET_OS_INFO_CMD lsb_release -a)
execute_process(COMMAND ${GET_OS_INFO_CMD} OUTPUT_VARIABLE OS_PROPERTIES OUTPUT_STRIP_TRAILING_WHITESPACE)
message("========= DETECTED OS =========")
message("${OS_PROPERTIES}")
message("===============================")

include(CheckLanguage)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
else()
    message(FATAL_ERROR "No CUDA compiler found")
endif()

set(CMAKE_CXX_VERSION 14)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_VERSION})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD ${CMAKE_CXX_VERSION})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

###################
### Setup flags ###
###################
message("CUDA-compiler: ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
message("CUDA-compiler location: ${CMAKE_CUDA_COMPILER}")
message("===============================")

#############################
### Library flags         ###
#############################
set(LIBS_OPTIONS -lcurand -lcudadevrt -lcudart -lculibos -lrt)

message(STATUS "LIBS    flags: ${LIBS_OPTIONS}")
message(STATUS "NVCC    flags: ${CUDA_OPTIONS}")
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "RELEASE flags: ${RELEASE_OPTIONS}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "DEBUG flags: ${DEBUG_OPTIONS}")
endif()
message("===============================")
message("Employing buildtype: ${CMAKE_BUILD_TYPE}")
message("===============================")


######################################################
### Add headers, sources, and wrappers recursively ###
######################################################
# include_directories("${CMAKE_CURRENT_SOURCE_DIR}/demaglibrary/include/headers")
# include_directories("${CMAKE_CURRENT_SOURCE_DIR}/python_wrapper")

file(GLOB_RECURSE SOURCES "sources/*.cpp")
file(GLOB_RECURSE CUSOURCES "sources/*.cu")
file(GLOB_RECURSE HEADERS "sources/*.h")

find_package(pybind11 REQUIRED)
pybind11_add_module(${PROJECT_NAME} ${SOURCES} ${CUSOURCES} ${HEADERS} ${PYTHON})

###########################################
###  Apply RELEASE/DEBUG compile flags  ###
###########################################
target_compile_options (${PROJECT_NAME} PRIVATE ${CUDA_OPTIONS})
target_compile_options (${PROJECT_NAME} PRIVATE ${LIBS_OPTIONS})
target_compile_options (${PROJECT_NAME} PRIVATE "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")
target_compile_options (${PROJECT_NAME} PRIVATE "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>")

target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBS_OPTIONS})
