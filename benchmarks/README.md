# Benchmarks

Sparse operator benchmarks for `mlx-lattice`.

The main entry point is:

```sh
uv run python benchmarks/bench_sparse_ops.py
```

Useful options:

```sh
uv run python benchmarks/bench_sparse_ops.py --scales 1000 5000 25000 100000
uv run python benchmarks/bench_sparse_ops.py --backend cpu
uv run python benchmarks/bench_sparse_ops.py --backend metal
uv run python benchmarks/bench_sparse_ops.py --backend cuda
uv run python benchmarks/bench_sparse_ops.py --warmup 3 --repeat 9 --csv benchmarks/results/sparse_ops.csv
```

The benchmark separates cold paths, which include coordinate map construction,
from hot paths, which reuse already-built maps. CPU runs use CPU arrays and CPU
coordinate maps. Metal and CUDA runs use MLX GPU arrays and the native GPU
coordinate/conv paths where implemented.
