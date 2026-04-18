"""Pytest hooks — prints the memory-strategy metrics table at end of session
and writes it to evals/out/metrics.md.

The counters themselves live in `evals/_metrics.py` so test modules can
import them without relying on pytest's conftest-loading semantics.
"""

from __future__ import annotations

from pathlib import Path

from ._metrics import METRICS, render_table


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    if not METRICS:
        return

    table = render_table()
    if not table:
        return

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_sep("=", "memory strategy summary")
        reporter.write_line(table)
    else:
        print("\n\n" + table + "\n")

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.md").write_text("```\n" + table + "\n```\n")
