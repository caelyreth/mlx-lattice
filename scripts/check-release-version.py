from __future__ import annotations

import os
import pathlib
import sys
import tomllib


def project_version(path: str) -> str:
    data = tomllib.loads(pathlib.Path(path).read_text())
    return str(data['project']['version'])


tag = os.environ.get('GITHUB_REF_NAME', '')
if tag.startswith('v'):
    tag = tag[1:]

versions = {
    'mlx-lattice': project_version('pyproject.toml'),
    'mlx-lattice-cuda13': project_version(
        'src/mlx_lattice_cuda13/pyproject.toml'
    ),
}

mismatches = {
    name: version for name, version in versions.items() if version != tag
}
if mismatches:
    for name, version in mismatches.items():
        print(
            f'{name} version {version} does not match release tag {tag}',
            file=sys.stderr,
        )
    raise SystemExit(1)
