include(CheckLanguage)

check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "MLX_LATTICE_BUILD_CUDA=ON requires nvcc.")
endif()

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(MLX_LATTICE_CUDA_ARTIFACT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/artifacts")
file(MAKE_DIRECTORY "${MLX_LATTICE_CUDA_ARTIFACT_DIR}")

function(mlx_lattice_cuda_ptx name source)
  set(output "${MLX_LATTICE_CUDA_ARTIFACT_DIR}/${name}.ptx")
  add_custom_command(
    OUTPUT "${output}"
    COMMAND
      "${CMAKE_CUDA_COMPILER}"
      --ptx
      --std=c++17
      --expt-relaxed-constexpr
      -I"${PROJECT_SOURCE_DIR}/native"
      "${source}"
      -o "${output}"
    DEPENDS "${source}" ${ARGN}
    VERBATIM
    COMMENT "Compiling CUDA PTX artifact ${name}.ptx")
  set(MLX_LATTICE_CUDA_ARTIFACTS
      ${MLX_LATTICE_CUDA_ARTIFACTS} "${output}"
      PARENT_SCOPE)
endfunction()

mlx_lattice_cuda_ptx(
  coords
  "${CMAKE_CURRENT_LIST_DIR}/coords/kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/coords/kernels.cuh")
mlx_lattice_cuda_ptx(
  conv
  "${CMAKE_CURRENT_LIST_DIR}/conv/kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/conv/kernels.cuh")
mlx_lattice_cuda_ptx(
  pool
  "${CMAKE_CURRENT_LIST_DIR}/pool/kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/pool/kernels.cuh")

add_custom_target(
  mlx_lattice_cuda_artifacts ALL
  DEPENDS ${MLX_LATTICE_CUDA_ARTIFACTS})

install(
  FILES ${MLX_LATTICE_CUDA_ARTIFACTS}
  DESTINATION "${MLX_LATTICE_CUDA_ARTIFACT_DESTINATION}")
