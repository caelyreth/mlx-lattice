from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from lattice_conformance.gameleon_geometry import (
    GeometryProfileWeights,
    Level8GeometryProfile,
    read_ascii_ply_xyz,
)
from lattice_conformance.replay import replay_fixtures


def main() -> None:
    args = _parse_args()
    if args.command == 'replay':
        report = replay_fixtures(Path(args.path), fail_fast=args.fail_fast)
        if args.report:
            Path(args.report).write_text(
                json.dumps(report.to_json(), indent=2) + '\n',
                encoding='utf-8',
            )
        print(json.dumps(report.to_json(), indent=2))
        if report.failures:
            raise SystemExit(1)
        return
    if args.command == 'geometry-profile':
        _run_geometry_profile(args)
        return
    raise AssertionError(args.command)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Cross-runtime lattice artifact conformance tools.'
    )
    subcommands = parser.add_subparsers(dest='command', required=True)
    replay = subcommands.add_parser(
        'replay',
        help='Replay Torch CUDA generated lattice fixtures on MLX.',
    )
    replay.add_argument(
        'path', help='Fixture directory or .tar.gz archive.'
    )
    replay.add_argument('--fail-fast', action='store_true')
    replay.add_argument(
        '--report',
        help='Optional JSON report path for per-case replay accuracy metrics.',
    )

    geometry = subcommands.add_parser(
        'geometry-profile',
        help='Replay the fixed Level-8 geometry BPP profile on MLX.',
    )
    geometry.add_argument('--weights', type=Path, required=True)
    geometry.add_argument('--input', type=Path, required=True)
    geometry.add_argument(
        '--dtype', choices=('float32', 'float16'), default='float32'
    )
    geometry.add_argument(
        '--device', choices=('cpu', 'metal'), default='metal'
    )
    geometry.add_argument('--warmup', type=_non_negative, default=2)
    geometry.add_argument('--repeats', type=_positive, default=10)
    geometry.add_argument(
        '--report',
        type=Path,
        help='Optional JSON path for the timing and stage report.',
    )

    return parser.parse_args()


def _run_geometry_profile(args: argparse.Namespace) -> None:
    _select_device(args.device)
    dtype = mx.float16 if args.dtype == 'float16' else mx.float32
    weights = GeometryProfileWeights.from_safetensors(args.weights)
    profile = Level8GeometryProfile(weights, dtype=dtype)
    xyz = read_ascii_ply_xyz(args.input)
    timing = profile.benchmark(
        profile.from_xyz(xyz), warmup=args.warmup, repeats=args.repeats
    )
    payload = {
        'input': str(args.input),
        'weights': str(args.weights),
        'device': args.device,
        'dtype': args.dtype,
        'input_points': timing.result.input_points,
        'levels': list(timing.result.levels),
        'bpp': timing.bpp,
        'median_ms': timing.median_ms,
        'samples_ms': list(timing.samples_ms),
        'stages': [
            {
                'depth': stage.depth,
                'parent_points': stage.parent_points,
                'target_points': stage.target_points,
                'group1_points': stage.group1_points,
                'group2_points': stage.group2_points,
            }
            for stage in timing.result.stages
        ],
    }
    if args.report:
        args.report.write_text(
            json.dumps(payload, indent=2) + '\n', encoding='utf-8'
        )
    print(json.dumps(payload, indent=2))


def _select_device(name: str) -> None:
    if name == 'cpu':
        mx.set_default_device(mx.Device(mx.cpu))
        return
    if not mx.metal.is_available():
        raise RuntimeError('Metal is unavailable in this MLX installation.')
    mx.set_default_device(mx.Device(mx.gpu))


def _non_negative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError('must be non-negative')
    return parsed


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return parsed


if __name__ == '__main__':
    main()
