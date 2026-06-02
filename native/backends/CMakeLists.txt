# MARK: - options

option(MLX_LATTICE_BUILD_METAL "Build mlx-lattice Metal coordinate backend" ON)

add_library(mlx_lattice_backends INTERFACE)

set(MLX_LATTICE_HAS_METAL 0)
if(APPLE)
  execute_process(
    COMMAND xcrun -sdk macosx -find metal
    RESULT_VARIABLE MLX_LATTICE_METAL_FIND_RESULT
    OUTPUT_QUIET
    ERROR_QUIET)

  if(MLX_BUILD_METAL
     AND MLX_LATTICE_BUILD_METAL
     AND MLX_LATTICE_METAL_FIND_RESULT EQUAL 0)
    set(MLX_LATTICE_HAS_METAL 1)
  endif()
endif()

target_compile_definitions(
  mlx_lattice_backends
  INTERFACE
    MLX_LATTICE_HAS_CPU=1
    MLX_LATTICE_HAS_METAL=${MLX_LATTICE_HAS_METAL}
    MLX_LATTICE_HAS_CUDA=0
    MLX_LATTICE_HAS_ROCM=0)

# MARK: - backends

add_subdirectory(cpu)
add_subdirectory(metal)
add_subdirectory(cuda)

target_link_libraries(
  mlx_lattice_backends
  INTERFACE
    mlx_lattice_backend_cpu
    mlx_lattice_backend_metal
    mlx_lattice_backend_cuda)
