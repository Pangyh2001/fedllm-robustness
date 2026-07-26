from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


START = "<!-- RESULTS_START -->"
END = "<!-- RESULTS_END -->"
SEED_SUFFIX = re.compile(r"__seed\d+$")
METRICS = ("clean", "robust", "bottom_tail", "worst_client", "client_std")


def is_smoke_run(summary_path: Path) -> bool:
    return any("smoke" in part.lower() for part in summary_path.parts)


def metric_values(result: dict) -> dict[str, float]:
    return {
        "clean": result["clean"]["client_macro"],
        "robust": result["robust"]["client_macro"],
        "bottom_tail": result["robust"]["bottom_tail"],
        "worst_client": result["robust"]["worst_client"],
        "client_std": result["robust"]["client_std"],
    }


def format_mean_std(values: list[float]) -> str:
    array = np.asarray(values, dtype=float)
    ddof = 1 if len(array) > 1 else 0
    return f"{array.mean():.4f} ± {array.std(ddof=ddof):.4f}"


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate formal runs by method/attack and update EXPERIMENTS.md."
    )
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--report", default="EXPERIMENTS.md")
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke runs. Disabled by default because smoke results are not evidence.",
    )
    args = parser.parse_args()

    groups: dict[tuple[str, str, str], list[tuple[int, dict[str, float]]]] = (
        defaultdict(list)
    )
    for summary_path in sorted(Path(args.outputs).rglob("summary.json")):
        if not args.include_smoke and is_smoke_run(summary_path):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for attack, result in (summary.get("final_test") or {}).items():
            run_group = SEED_SUFFIX.sub("", summary_path.parent.name)
            key = (run_group, summary["algorithm"], attack)
            groups[key].append((int(summary["seed"]), metric_values(result)))

    rows = []
    for (run_group, algorithm, attack), records in sorted(groups.items()):
        seeds = sorted(seed for seed, _ in records)
        values = {
            metric: [record[metric] for _, record in records] for metric in METRICS
        }
        rows.append(
            [
                run_group,
                algorithm,
                attack.upper(),
                str(len(seeds)),
                ",".join(map(str, seeds)),
                *(format_mean_std(values[metric]) for metric in METRICS),
            ]
        )

    lines = [
        START,
        "",
        "| Run group | Method | Attack | n | Seeds | Clean macro | Robust macro | Bottom-20% | Worst client | Client std |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    if rows:
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
    else:
        lines.append("| 尚未运行 | — | — | — | — | — | — | — | — | — |")
    lines.extend(["", END])

    report = Path(args.report)
    content = report.read_text(encoding="utf-8")
    if content.count(START) != 1 or content.count(END) != 1:
        raise RuntimeError(f"{report} 中的结果表标记缺失或重复")
    before, remainder = content.split(START, 1)
    _, after = remainder.split(END, 1)
    report.write_text(before + "\n".join(lines) + after, encoding="utf-8")
    print(f"updated {report}: {len(rows)} grouped rows")


if __name__ == "__main__":
    main()
