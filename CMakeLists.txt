cmake_minimum_required(VERSION 3.20)
project(ALP_Compare LANGUAGES CXX CUDA)

# ===== 语言标准 =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ===== CUDA 架构（RTX 4090 → sm_89）=====
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# Release 缺省
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ===== 头文件路径 =====
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ===== 源文件 =====
set(SRC
  src/compare_main.cpp
  src/gpu/alp_gpu.cu
)

add_executable(alp_compare ${SRC})
set_target_properties(alp_compare PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 避免某些环境下的 nvcc 冗余警告
if (MSVC)
  target_compile_definitions(alp_compare PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(alp_compare PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O3 -v>)
endif()

# ===== 运行后便捷命令 =====
add_custom_target(run
  COMMAND $<TARGET_FILE:alp_compare>
  DEPENDS alp_compare
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running alp_compare"
)