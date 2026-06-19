# MARK: - options

option(MLX_LATTICE_BUILD_METAL "Build mlx-lattice Metal coordinate backend" ON)
option(MLX_LATTICE_BUILD_CUDA "Build mlx-lattice CUDA backend" OFF)

add_library(mlx_lattice_backends INTERFACE)

set(MLX_LATTICE_HAS_METAL 0)
set(MLX_LATTICE_HAS_CUDA 0)
if(APPLE)
  cmake_host_system_information(
    RESULT MLX_LATTICE_MACOS_VERSION QUERY OS_RELEASE)
  string(REGEX MATCH "^[0-9]+" MLX_LATTICE_MACOS_MAJOR
               "${MLX_LATTICE_MACOS_VERSION}")

  execute_process(
    COMMAND xcrun -sdk macosx -find metal
    RESULT_VARIABLE MLX_LATTICE_METAL_FIND_RESULT
    OUTPUT_QUIET
    ERROR_QUIET)

  if(MLX_BUILD_METAL
     AND MLX_LATTICE_BUILD_METAL
     AND MLX_LATTICE_MACOS_MAJOR GREATER_EQUAL 25
     AND MLX_LATTICE_METAL_FIND_RESULT EQUAL 0)
    set(MLX_LATTICE_HAS_METAL 1)
  endif()
endif()

if(APPLE AND MLX_LATTICE_BUILD_CUDA)
  message(FATAL_ERROR "MLX_LATTICE_BUILD_CUDA is only supported on Linux.")
endif()

if(NOT APPLE AND MLX_LATTICE_BUILD_CUDA)
  include(CheckLanguage)
  include(CheckIncludeFileCXX)
  find_package(CUDAToolkit QUIET)
  set(CMAKE_REQUIRED_INCLUDES ${MLX_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
  check_include_file_cxx("mlx/backend/cuda/cuda.h" MLX_LATTICE_HAS_MLX_CUDA)
  check_include_file_cxx("cudnn.h" MLX_LATTICE_HAS_CUDNN_HEADER)
  check_include_file_cxx("fmt/format.h" MLX_LATTICE_HAS_FMT_HEADER)
  unset(CMAKE_REQUIRED_INCLUDES)
  check_language(CUDA)

  if(MLX_LATTICE_HAS_MLX_CUDA
     AND MLX_LATTICE_HAS_CUDNN_HEADER
     AND MLX_LATTICE_HAS_FMT_HEADER
     AND CUDAToolkit_FOUND
     AND CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES
       OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
      set(CMAKE_CUDA_ARCHITECTURES
          native
          CACHE STRING "CUDA architectures for mlx-lattice kernels" FORCE)
    endif()
    enable_language(CUDA)
    set(MLX_LATTICE_HAS_CUDA 1)
  else()
    message(
      FATAL_ERROR
        "MLX_LATTICE_BUILD_CUDA=ON requires MLX CUDA headers, CUDAToolkit, "
        "CUDNN headers, fmt headers, and a CUDA compiler. Install "
        "mlx[cuda12] or mlx[cuda13], libcudnn development headers, fmt "
        "development headers, and ensure nvcc is available."
    )
  endif()
endif()

target_compile_definitions(
  mlx_lattice_backends
  INTERFACE
    MLX_LATTICE_HAS_CPU=1
    MLX_LATTICE_HAS_METAL=${MLX_LATTICE_HAS_METAL}
    MLX_LATTICE_HAS_CUDA=${MLX_LATTICE_HAS_CUDA})

# MARK: - backends

add_subdirectory(cpu)
add_subdirectory(metal)
add_subdirectory(cuda)

add_library(mlx_lattice_backend_gpu STATIC gpu.cpp)

target_include_directories(
  mlx_lattice_backend_gpu
  PUBLIC ${PROJECT_SOURCE_DIR}/native)

target_compile_definitions(
  mlx_lattice_backend_gpu
  PUBLIC
    MLX_LATTICE_HAS_CPU=1
    MLX_LATTICE_HAS_METAL=${MLX_LATTICE_HAS_METAL}
    MLX_LATTICE_HAS_CUDA=${MLX_LATTICE_HAS_CUDA})

target_link_libraries(
  mlx_lattice_backend_gpu
  PUBLIC
    mlx_lattice_backend_cpu
    mlx_lattice_backend_metal
    mlx_lattice_backend_cuda)

target_link_libraries(
  mlx_lattice_backends
  INTERFACE
    mlx_lattice_backend_cpu
    mlx_lattice_backend_metal
    mlx_lattice_backend_cuda
    mlx_lattice_backend_gpu)
