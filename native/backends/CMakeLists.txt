# MARK: - options

option(MLX_LATTICE_BUILD_METAL "Build mlx-lattice Metal coordinate backend" ON)
option(
  MLX_LATTICE_BUILD_CUDA
  "Build mlx-lattice precompiled CUDA kernel artifacts"
  OFF)
set(
  MLX_LATTICE_CUDA_ARTIFACT_DESTINATION
  "mlx_lattice/backends/cuda/artifacts"
  CACHE STRING "Python package directory for compiled CUDA artifacts")

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

target_compile_definitions(
  mlx_lattice_backends
  INTERFACE
    MLX_LATTICE_HAS_CPU=1
    MLX_LATTICE_HAS_METAL=${MLX_LATTICE_HAS_METAL}
    MLX_LATTICE_HAS_CUDA=${MLX_LATTICE_HAS_CUDA})

# MARK: - backends

add_subdirectory(cpu)
add_subdirectory(metal)
if(MLX_LATTICE_BUILD_CUDA AND NOT APPLE)
  add_subdirectory(cuda)
endif()

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
    mlx_lattice_backend_metal)

target_link_libraries(
  mlx_lattice_backends
  INTERFACE
    mlx_lattice_backend_cpu
    mlx_lattice_backend_metal
    mlx_lattice_backend_gpu)
