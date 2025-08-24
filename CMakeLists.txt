cmake_minimum_required(VERSION 3.20)
project(ALP LANGUAGES CXX CUDA)

# C++/CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 默认 Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 可按需修改计算能力（4090：89）
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(CUDAToolkit REQUIRED)

# 头文件路径（对内对外都可见）
set(ALP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ========= GPU 静态库 =========
# 修改 src/alp_gpu.cu 为你的真实路径
add_library(alp_gpu STATIC
  src/alp_gpu.cu
)

target_include_directories(alp_gpu
  PUBLIC
    $<BUILD_INTERFACE:${ALP_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(alp_gpu
  PUBLIC CUDA::cudart
)

set_target_properties(alp_gpu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

add_executable(alp_compare
  examples/compare_main.cpp
)

target_include_directories(alp_compare
  PRIVATE ${ALP_INCLUDE_DIR}
)

target_link_libraries(alp_compare
  PRIVATE alp_gpu
)
