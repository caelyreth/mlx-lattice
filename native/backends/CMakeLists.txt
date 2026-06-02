# MARK: - registry

add_library(mlx_lattice_backends INTERFACE)

target_compile_definitions(
  mlx_lattice_backends
  INTERFACE
    MLX_LATTICE_HAS_CPU=1
    MLX_LATTICE_HAS_METAL=0
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
