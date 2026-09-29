import argparse
from collections.abc import Sequence

from agentipc import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the AgentIPC command-line parser."""
    parser = argparse.ArgumentParser(prog="agentipc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="Show the AgentIPC version")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run offline environment checks",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Emit one machine-readable JSON document",
    )

    demo_parser = subparsers.add_parser(
        "demo",
        help="Run the offline A/B/D AgentIPC demo",
    )
    demo_parser.add_argument(
        "--provider",
        choices=("mock",),
        required=True,
        help="Provider to use for the demo",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AgentIPC command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0

    if args.command == "doctor":
        from agentipc.doctor import render_doctor_human, render_doctor_json, run_doctor

        report = run_doctor()
        if args.json_output:
            print(render_doctor_json(report))
        else:
            print(render_doctor_human(report))
        return 0 if report.ok else 1

    if args.command == "demo":
        from agentipc.demo import render_demo, run_demo

        report = run_demo(provider=args.provider)
        print(render_demo(report))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2
