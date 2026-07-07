#!/usr/bin/env bash
set -euo pipefail

tool="${1:-build/clangd-mlir/mlir/tools/lattice-opt/lattice-opt}"

for fixture in mlir/test/Dialect/Lattice/valid/*.mlir; do
  "${tool}" "${fixture}" -o /tmp/lattice-valid.mlir >/tmp/lattice-valid.out
done

for fixture in mlir/test/Dialect/Lattice/invalid/*.mlir; do
  if "${tool}" "${fixture}" -o /tmp/lattice-invalid.mlir \
    >/tmp/lattice-invalid.out 2>/tmp/lattice-invalid.err; then
    echo "expected invalid fixture to fail: ${fixture}" >&2
    exit 1
  fi
done
