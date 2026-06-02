# MARK: - target

add_library(mlx_lattice_core STATIC)

target_sources(
  mlx_lattice_core
  PRIVATE
    arithmetic.cpp
    runtime.cpp)

target_include_directories(
  mlx_lattice_core
  PUBLIC ${PROJECT_SOURCE_DIR}/native)

target_link_libraries(
  mlx_lattice_core
  PUBLIC
    mlx_lattice_config
    mlx_lattice_backends)
