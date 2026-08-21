from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from fedrda_experiments.config import load_config
from fedrda_experiments.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "resolved_config.json")
    if args.model_name_or_path:
        cfg.model.name_or_path = args.model_name_or_path

    runner = ExperimentRunner(cfg, resume=True)
    runner.run_dir = run_dir
    runner.checkpoint_path = run_dir / "latest_checkpoint.pt"
    runner.metrics_path = run_dir / "round_metrics.jsonl"
    runner._prepare()
    runner.eval_attack_cfg = dataclasses.replace(
        cfg.attack,
        epsilon=(
            cfg.attack.eval_epsilon
            if cfg.attack.eval_epsilon is not None
            else cfg.attack.epsilon
        ),
    )
    metrics = runner._evaluate_test(final_evaluation=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
