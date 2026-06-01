#!/usr/bin/env bash
set -euo pipefail

expected_count="${1:?usage: scripts/collect-distributions.sh <expected-count> [source-dir] [output-dir]}"
source_dir="${2:-artifacts}"
out_dir="${3:-dist}"

rm -rf "${out_dir}"
mkdir -p "${out_dir}"

shopt -s globstar nullglob
declare -A seen

for file in "${source_dir}"/**/*; do
  if [[ ! -f "${file}" ]]; then
    continue
  fi

  case "${file}" in
    *.whl|*.tar.gz) ;;
    *) continue ;;
  esac

  filename="$(basename "${file}")"
  if [[ -n "${seen[$filename]:-}" ]]; then
    echo "::error::Duplicate distribution filename: ${filename}"
    echo "::error::First: ${seen[$filename]}"
    echo "::error::Second: ${file}"
    exit 1
  fi

  seen["${filename}"]="${file}"
  cp "${file}" "${out_dir}/${filename}"
done

if [[ "${#seen[@]}" -ne "${expected_count}" ]]; then
  echo "::error::Expected ${expected_count} distributions, found ${#seen[@]}"
  find "${source_dir}" -type f -print
  exit 1
fi
