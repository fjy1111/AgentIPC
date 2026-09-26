import argparse
from collections.abc import Sequence

from agentipc import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal AgentIPC command-line parser."""
    parser = argparse.ArgumentParser(prog="agentipc")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("version", help="Show the AgentIPC version")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AgentIPC command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2