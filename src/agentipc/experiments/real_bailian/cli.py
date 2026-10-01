from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from agentipc.experiments.real_bailian.config import (
    RealBailianConfigError,
    load_real_bailian_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_real_bailian.py",
        description="AgentIPC paid real-Bailian experiment harness",
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)
    calibration = subparsers.add_parser("calibration")
    calibration.add_argument(
        "--confirm-real-api",
        action="store_true",
        help="Explicitly authorize real network/API calls that may incur cost",
    )
    calibration.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="AgentIPC repository root (default: current directory)",
    )
    calibration.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Optional result root; defaults to <repo>/results",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.phase == "calibration" and not args.confirm_real_api:
        print("Real API execution requires --confirm-real-api")
        return 2

    env = os.environ if environ is None else environ
    try:
        config = load_real_bailian_config(env)
    except (RealBailianConfigError, TypeError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    # Import the runner only after the explicit paid-API gate and safe config load.
    from agentipc.experiments.real_bailian.runner import run_calibration

    try:
        result_dir, summary = run_calibration(
            config=config,
            repo_root=args.repo_root,
            results_root=args.results_root,
        )
    except FileExistsError as exc:
        print(f"Result directory already exists: {exc.filename}", file=sys.stderr)
        return 2
    except Exception as exc:
        message = str(exc)
        for secret in (config.api_key, config.base_url):
            if secret:
                message = message.replace(secret, "<redacted>")
        print(f"Calibration failed before report completion: {exc.__class__.__name__}: {message}", file=sys.stderr)
        return 1

    print(f"Calibration results: {result_dir}")
    print("PASS" if summary.get("pass") is True else "FAIL")
    return 0 if summary.get("pass") is True else 1
