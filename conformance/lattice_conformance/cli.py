from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    return parser.parse_args()


if __name__ == '__main__':
    main()
