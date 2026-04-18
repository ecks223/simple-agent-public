"""Per-turn pass/fail tally for the memory-strategy harness.

Kept in a plain module (not conftest.py) so both the hook and the test module
import from the same place — pytest imports conftest with special semantics,
and test files importing from conftest is finicky across pytest versions.
"""

from __future__ import annotations

from collections import defaultdict

# METRICS[strategy_name][category] -> list[bool] (one bool per turn)
METRICS: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))


def record_turn(strategy_name: str, category: str, passed: bool) -> None:
    METRICS[strategy_name][category].append(passed)


def render_table() -> str:
    if not METRICS:
        return ""

    categories = sorted({c for by_cat in METRICS.values() for c in by_cat})
    strategies = list(METRICS.keys())

    header = ["strategy", *categories, "overall"]
    rows: list[list[str]] = []
    for strat in strategies:
        by_cat = METRICS[strat]
        cells = [strat]
        total_pass = 0
        total_count = 0
        for cat in categories:
            results = by_cat.get(cat, [])
            passed = sum(results)
            count = len(results)
            cells.append(f"{passed}/{count}" if count else "-")
            total_pass += passed
            total_count += count
        if total_count:
            cells.append(
                f"{total_pass}/{total_count} "
                f"({100 * total_pass / total_count:.0f}%)"
            )
        else:
            cells.append("-")
        rows.append(cells)

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    divider = "  ".join("-" * w for w in widths)

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    return "\n".join(
        [
            "Memory Strategy Evaluation",
            "=" * max(len(divider), 24),
            fmt(header),
            divider,
            *(fmt(r) for r in rows),
        ]
    )
