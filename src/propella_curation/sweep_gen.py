"""Generate a sweep of scoring-config YAMLs from a single sweep spec.

A *sweep spec* (YAML) describes one experiment family: a base quality ladder
(ordered tiers + shape), an optional hard floor, a list of temperatures, and
the shared sampling/run parameters. From it this generator emits:

  * one **scoring-config YAML per temperature** (the only field that varies),
  * a **manifest.json** recording every run's full parameters (for the
    analysis step), and
  * a **submit_all.sh** that launches every run via ``submit_curation.sh``.

The base ladder fixes only the *ordering and meaning* of the tiers; the
temperature β is the single knob that controls how hard selection leans on
that ordering (``P(select i) ∝ score_i ** β``). See ``docs/math.md``.

Usage:
    propella-gen-sweep path/to/sweep_spec.yaml
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# ----------------------------------------------------------------------------
# Base ladder construction
# ----------------------------------------------------------------------------
def base_category_scores(
    tiers: list[str], shape: str, floor: list[str]
) -> dict[str, float]:
    """Map ordered tiers (low → high) to base scores S for the given shape.

    Floored tiers get a hard 0 (disqualified at every temperature, since the
    sampler masks zeros). Non-floored tiers keep their position L on the ladder
    (flooring removes the bottom, it does not renumber the ladder):

      * ``geometric``: S = e^L / e^(K-1)  → constant per-step odds (e^β).
      * ``linear``:    S = (L + 1) / K    → equal-interval quality.
    """
    K = len(tiers)
    floor_set = set(floor)
    scores: dict[str, float] = {}
    for L, tier in enumerate(tiers):
        if tier in floor_set:
            scores[tier] = 0.0
        elif shape == "geometric":
            scores[tier] = round(math.exp(L) / math.exp(K - 1), 6)
        elif shape == "linear":
            scores[tier] = round((L + 1) / K, 6)
        else:
            raise ValueError(f"Unknown base_shape '{shape}' (use geometric|linear)")
    return scores


# ----------------------------------------------------------------------------
# Spec
# ----------------------------------------------------------------------------
@dataclass
class SweepSpec:
    name: str
    column: str
    tiers: list[str]
    base_shape: str
    floor: list[str]
    temperatures: list[float]
    # run parameters (shared across the sweep)
    mode: str
    dataset_path: str
    annotations_path: str
    output_root: str
    n_samples: int | None = None
    max_duplications: int | None = None
    seed: int = 42
    writer: str = "load"
    # scoring config knobs
    aggregation: str = "weighted_mean"
    normalize: bool = False
    missing_id_score: float = 0.0
    weight: float = 1.0
    sampling: str = "iid"  # 'iid' or 'systematic'
    # where to write the generated config YAMLs (relative to the spec file);
    # defaults to the derived run id (name + sampling [+ floored])
    config_out_dir: str | None = None
    # optional SLURM overrides forwarded to submit_curation.sh
    slurm: dict[str, str] | None = None

    @classmethod
    def from_file(cls, path: Path) -> "SweepSpec":
        raw = yaml.safe_load(path.read_text())
        known = cls.__dataclass_fields__.keys()
        unknown = set(raw) - set(known)
        if unknown:
            raise ValueError(f"Unknown spec keys: {sorted(unknown)}")
        return cls(**raw)


def _beta_tag(beta: float) -> str:
    """Compact, filesystem-safe tag for a temperature value (1.0 -> 'b1', 0.5 -> 'b0.5')."""
    return "b" + format(beta, "g")


_SAMPLING_TAG = {"iid": "iid", "systematic": "sys"}


def run_id(spec: "SweepSpec") -> str:
    """Self-describing id for a sweep: ``<name>_<sampling>[_floored]``.

    The sampling method (and floor) are baked into every output path so that an
    iid run can never silently overwrite a systematic one, and the directory
    name always states how the data was drawn.
    """
    rid = f"{spec.name}_{_SAMPLING_TAG.get(spec.sampling, spec.sampling)}"
    if spec.floor:
        rid += "_floored"
    return rid


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
def generate(spec_path: str | Path) -> dict[str, Any]:
    spec_path = Path(spec_path).resolve()
    spec = SweepSpec.from_file(spec_path)

    rid = run_id(spec)
    base_dir = spec_path.parent
    cfg_dir = (base_dir / (spec.config_out_dir or rid)).resolve()
    cfg_dir.mkdir(parents=True, exist_ok=True)

    cat_scores = base_category_scores(spec.tiers, spec.base_shape, spec.floor)

    runs = []
    for beta in spec.temperatures:
        tag = _beta_tag(beta)
        config = {
            "temperature": float(beta),
            "sampling": spec.sampling,
            "aggregation": spec.aggregation,
            "normalize": spec.normalize,
            "missing_id_score": spec.missing_id_score,
            "columns": {
                spec.column: {
                    "weight": spec.weight,
                    "default_score": 0.0,
                    "category_scores": cat_scores,
                }
            },
        }
        cfg_path = cfg_dir / f"{rid}_{tag}.yaml"
        header = (
            f"# Generated by propella-gen-sweep from {spec_path.name} — do not edit by hand.\n"
            f"# sweep={rid} shape={spec.base_shape} sampling={spec.sampling} "
            f"temperature={beta} floor={spec.floor or 'none'}\n"
        )
        cfg_path.write_text(header + yaml.safe_dump(config, sort_keys=False))

        runs.append(
            {
                "name": f"{rid}_{tag}",
                "temperature": float(beta),
                "config": str(cfg_path),
                "output_dir": f"{spec.output_root.rstrip('/')}/{rid}/{tag}",
                "mode": spec.mode,
                "n_samples": spec.n_samples,
                "max_duplications": spec.max_duplications,
                "seed": spec.seed,
                "writer": spec.writer,
            }
        )

    manifest = {
        "sweep": rid,
        "name": spec.name,
        "base_shape": spec.base_shape,
        "sampling": spec.sampling,
        "floor": spec.floor,
        "base_category_scores": cat_scores,
        "dataset_path": spec.dataset_path,
        "annotations_path": spec.annotations_path,
        "output_root": f"{spec.output_root.rstrip('/')}/{rid}",
        "runs": runs,
    }
    manifest_path = base_dir / f"{rid}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    submit_path = base_dir / f"{rid}.submit_all.sh"
    submit_path.write_text(_render_submit(spec, runs))
    submit_path.chmod(0o755)

    return {
        "spec": str(spec_path),
        "config_dir": str(cfg_dir),
        "manifest": str(manifest_path),
        "submit_all": str(submit_path),
        "n_runs": len(runs),
        "base_category_scores": cat_scores,
    }


def _render_submit(spec: SweepSpec, runs: list[dict]) -> str:
    slurm = spec.slurm or {}
    slurm_flags = "".join(
        f" --{k} {v}" for k, v in slurm.items()
    )  # e.g. --time 00:20:00 --mem 128G --cpus 8
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by propella-gen-sweep — launches every run in the sweep.",
        "set -euo pipefail",
        'SUBMIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)/submit_curation.sh"',
        "",
    ]
    for r in runs:
        cmd = [
            'bash "$SUBMIT"',
            f"--dataset-path {spec.dataset_path}",
            f"--annotations-path {spec.annotations_path}",
            f"--output-dir {r['output_dir']}",
            f"--mode {r['mode']}",
            f"--config {r['config']}",
            f"--seed {r['seed']}",
            f"--writer {r['writer']}",
        ]
        if r["n_samples"] is not None:
            cmd.append(f"--n-samples {r['n_samples']}")
        if r["max_duplications"] is not None:
            cmd.append(f"--max-duplications {r['max_duplications']}")
        if slurm_flags:
            cmd.append(slurm_flags.strip())
        lines.append(f"echo '>>> {r['name']} (β={r['temperature']})'")
        lines.append(" \\\n    ".join(cmd))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("spec", help="Path to the sweep spec YAML")
    args = parser.parse_args()
    result = generate(args.spec)
    print(f"Generated sweep '{Path(args.spec).stem}':")
    print(f"  configs:    {result['config_dir']}/ ({result['n_runs']} files)")
    print(f"  manifest:   {result['manifest']}")
    print(f"  submit_all: {result['submit_all']}")
    print(f"  base category_scores: {result['base_category_scores']}")
    print(f"\nLaunch with:  bash {result['submit_all']}")


if __name__ == "__main__":
    main()
