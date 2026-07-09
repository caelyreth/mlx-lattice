from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path


def fixture_root(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.suffixes[-2:] != ['.tar', '.gz']:
        raise ValueError(
            f'expected fixture directory or .tar.gz archive: {path}'
        )
    temp = Path(tempfile.mkdtemp(prefix='mlx_lattice_fuzz_'))
    with tarfile.open(path, 'r:gz') as archive:
        archive.extractall(temp, filter='data')
    roots = [item for item in temp.iterdir() if item.is_dir()]
    if len(roots) != 1:
        raise ValueError(f'archive must contain one root directory: {path}')
    return roots[0]


__all__ = ['fixture_root']
