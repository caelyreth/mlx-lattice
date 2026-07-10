from __future__ import annotations

import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def fixture_root(path: Path) -> Iterator[Path]:
    if path.is_dir():
        yield path
        return
    if path.suffixes[-2:] != ['.tar', '.gz']:
        raise ValueError(
            f'expected fixture directory or .tar.gz archive: {path}'
        )
    with tempfile.TemporaryDirectory(prefix='mlx_lattice_fuzz_') as temp:
        root = Path(temp)
        with tarfile.open(path, 'r:gz') as archive:
            archive.extractall(root, filter='data')
        roots = [item for item in root.iterdir() if item.is_dir()]
        if len(roots) != 1:
            raise ValueError(
                f'archive must contain one root directory: {path}'
            )
        yield roots[0]


__all__ = ['fixture_root']
