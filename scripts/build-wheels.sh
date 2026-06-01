#!/usr/bin/env bash
set -euo pipefail

target="${1:?usage: scripts/build-wheels.sh <macos|linux-cpu|linux-cuda13>}"
pythons="${MLX_LATTICE_PYTHONS:-3.12 3.13 3.14}"
out_dir="${MLX_LATTICE_WHEEL_DIR:-dist}"
manylinux_tag="${MLX_LATTICE_MANYLINUX_TAG:-manylinux_2_35_x86_64}"
cuda_architectures="${MLX_LATTICE_CUDA_ARCHITECTURES:-80}"

mkdir -p "${out_dir}"
uv python install ${pythons}

for python in ${pythons}; do
  echo "::group::Build ${target} wheel for Python ${python}"
  python_tag="cp${python/./}"

  case "${target}" in
    macos)
      uv build \
        --wheel \
        --python "${python}" \
        --out-dir "${out_dir}"
      ;;
    linux-cpu)
      rm -rf .venv
      UV_PYTHON="${python}" uv sync \
        --group dev \
        --no-install-project
      UV_PYTHON="${python}" uv build \
        --wheel \
        --python "${python}" \
        --no-build-isolation \
        --out-dir "${out_dir}" \
        -Cwheel.tags="${python_tag}-${python_tag}-${manylinux_tag}" \
        -Ccmake.define.MLX_LATTICE_BUILD_CUDA=OFF \
        -Ccmake.define.MLX_LATTICE_BUILD_METAL=OFF
      ;;
    linux-cuda13)
      rm -rf .venv
      UV_PYTHON="${python}" uv sync \
        --group dev \
        --group cuda-build \
        --no-install-project
      UV_PYTHON="${python}" uv build \
        src/mlx_lattice_cuda13 \
        --wheel \
        --python "${python}" \
        --no-build-isolation \
        --out-dir "${out_dir}" \
        -Cwheel.tags="${python_tag}-${python_tag}-${manylinux_tag}" \
        -Ccmake.define.CMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
        -Ccmake.define.MLX_LATTICE_REQUIRE_CUDA=ON
      ;;
    *)
      echo "Unknown wheel target: ${target}" >&2
      exit 1
      ;;
  esac

  echo "::endgroup::"
done
