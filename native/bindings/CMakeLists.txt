# MARK: - target

nanobind_add_module(
  _ext
  STABLE_ABI
  NB_STATIC
  module.cpp)

target_link_libraries(_ext PRIVATE mlx_lattice_core)

# MARK: - install

install(TARGETS _ext LIBRARY DESTINATION mlx_lattice)
