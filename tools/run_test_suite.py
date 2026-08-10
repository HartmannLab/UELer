#!/usr/bin/env python3
"""Run UELer's unittest suite and fail when tests are silently skipped.

``python -m unittest discover tests`` reports skips as a parenthesised count on
a line that is easy to scroll past: an environment without bokeh drops ~14
tests and still prints ``OK``. That is how the Python 3.11 coverage gap found in
issue #79 stayed invisible until the release audit — the suite was "passing" on
an interpreter where the whole bokeh/histogram path never ran.

In a complete environment the suite skips nothing, so CI runs this with
``--max-skips 0``. Every skip is printed with its reason before the exit code is
decided, so a failure says which dependency is missing rather than just "too
many skips".

Run from the repository root::

    python tools/run_test_suite.py                 # gate at zero skips
    python tools/run_test_suite.py --max-skips 20  # tolerate a partial env
"""

from __future__ import annotations

import argparse
import sys
import unittest


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--start-dir",
        default="tests",
        help="directory to discover tests in (default: tests)",
    )
    parser.add_argument(
        "--top-level-dir",
        default=".",
        help="import root for the discovered tests (default: the current directory)",
    )
    parser.add_argument(
        "--pattern",
        default="test*.py",
        help="test file glob (default: test*.py)",
    )
    parser.add_argument(
        "--max-skips",
        type=int,
        default=0,
        help=(
            "fail if more than this many tests are skipped (default: 0). "
            "Raise it only as a deliberate, documented allowance."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        dest="verbosity",
        help="increase unittest verbosity (repeatable)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # discover() puts top_level_dir on sys.path itself, so this works when the
    # script is invoked as tools/run_test_suite.py from the repository root.
    suite = unittest.defaultTestLoader.discover(
        args.start_dir,
        pattern=args.pattern,
        top_level_dir=args.top_level_dir,
    )
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)

    if result.skipped:
        print(f"\n{len(result.skipped)} skipped test(s):", file=sys.stderr)
        for test, reason in result.skipped:
            print(f"  {test.id()}\n      {reason}", file=sys.stderr)
    else:
        print("\nNo tests were skipped.", file=sys.stderr)

    if not result.wasSuccessful():
        return 1

    if len(result.skipped) > args.max_skips:
        print(
            f"\nFAIL: {len(result.skipped)} skipped test(s) exceeds the allowed "
            f"maximum of {args.max_skips}. A skipped test is an untested code "
            f"path, not a passing one — install the missing dependency, or raise "
            f"--max-skips deliberately and say why.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
