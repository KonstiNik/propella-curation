"""Generate a HuggingFace-compatible dataset card for curated datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

# At most this many distinct composite-score values -> show a composition table.
# Above it the score is treated as continuous (percentile summary).
_DISCRETE_MAX = 20

_TEMPLATE = """\
---
tags:
- propella-curated
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# {name} (propella-curated)

Curated subset of `{source_dataset}`, filtered using
[propella-curation](https://github.com/KonstiNik/propella-curation).

## Curation details

| Parameter | Value |
|-----------|-------|
| Source dataset | `{source_dataset}` |
| Annotations | `{annotations_path}` |
| Config | {config_name} |
| Mode | {mode_description} |
| Temperature (β) | {temperature} |
| Sampling | {sampling} |
| Source rows | {source_rows:,} |
| Selected rows | {selected_rows:,} ({selection_pct:.1f}%) |
| Seed | {seed} |
| Writer | {writer} |
| Date | {date} |

## Quality composition (before → after)

{composition}

## Coverage / resampling

{resampling}

## Source dataset

See the original dataset at `{source_dataset}` for full details
on content, sources, licensing, and citation.
"""


@dataclass
class CurationInfo:
    """All metadata needed to generate a dataset card."""

    name: str
    source_dataset: str
    annotations_path: str
    config_name: str
    mode: str
    threshold: float | None
    n_samples: int | None
    seed: int
    source_rows: int
    selected_rows: int
    scores_after: "numpy.ndarray"  # noqa: F821  score of each selected row (w/ dups)
    scores_before: "numpy.ndarray"  # noqa: F821  score of every source row
    gather_indices: "numpy.ndarray"  # noqa: F821  selected source-row positions
    max_duplications: int | None = None
    writer: str = "load"
    temperature: float = 1.0
    sampling: str = "iid"
    # score value -> tier label, when the mapping is unambiguous (else numeric)
    score_labels: dict[float, str] | None = None


def _composition_section(info: CurationInfo) -> str:
    """Before→after composition: a labelled table for discrete scores, else a
    percentile summary (the value-adaptive switch)."""
    import numpy as np

    sb = np.round(np.asarray(info.scores_before, dtype=float), 6)
    sa = np.round(np.asarray(info.scores_after, dtype=float), 6)
    if len(sa) == 0:
        return "_No rows selected._"

    mean_line = f"\n\nMean score: {sb.mean():.3f} → {sa.mean():.3f}"
    distinct = np.unique(sb)

    if len(distinct) <= _DISCRETE_MAX:
        labels = info.score_labels or {}
        nb, na = len(sb), len(sa)
        rows = ["| tier / score | before | after |", "|---|---|---|"]
        for v in sorted(distinct.tolist(), reverse=True):
            b = int((sb == v).sum())
            a = int((sa == v).sum())
            name = labels.get(v, format(v, "g"))
            rows.append(
                f"| {name} | {b:,} ({b / nb * 100:.1f}%) | {a:,} ({a / na * 100:.1f}%) |"
            )
        return "\n".join(rows) + mean_line

    # continuous fallback: percentile summary of the selected scores
    pcts = np.percentile(sa, [10, 25, 50, 75, 90])
    return (
        "```\n"
        f"Mean={sa.mean():.3f}  Median={np.median(sa):.3f}  Std={sa.std():.3f}\n"
        f"Min={sa.min():.3f}  Max={sa.max():.3f}\n"
        f"p10={pcts[0]:.3f}  p25={pcts[1]:.3f}  p50={pcts[2]:.3f}  "
        f"p75={pcts[3]:.3f}  p90={pcts[4]:.3f}\n"
        "```" + mean_line
    )


def _resampling_section(info: CurationInfo) -> str:
    """Coverage / duplication profile, computed from the gather indices."""
    import numpy as np

    gi = np.asarray(info.gather_indices)
    n_out = len(gi)
    src = info.source_rows
    if n_out == 0:
        return "_No rows selected._"

    _, counts = np.unique(gi, return_counts=True)
    uniq = len(counts)
    dropped = src - uniq
    maxc = int(counts.max())

    cap = 5
    parts = []
    for k in range(1, cap + 1):
        c = int((counts == k).sum())
        if c:
            parts.append(f"{k}×{c:,}")
    tail = int((counts > cap).sum())
    if tail:
        parts.append(f"{cap + 1}+×{tail:,}")

    def pct(x: int) -> float:
        return x / src * 100 if src else 0.0

    return (
        "```\n"
        f"rows out:            {n_out:,}\n"
        f"unique source rows:  {uniq:,} ({pct(uniq):.1f}% of source)\n"
        f"dropped source rows: {dropped:,} ({pct(dropped):.1f}% of source)\n"
        f"max copies of a row: {maxc}\n"
        f"copies distribution: {'  '.join(parts)}\n"
        "```"
    )


def write_dataset_card(info: CurationInfo, output_dir: str) -> None:
    """Write a README.md dataset card to the output directory."""
    # n_samples is None when the user didn't pass --n_samples; apply() then
    # defaults it to len(dataset), which equals source_rows.
    n_display = info.n_samples if info.n_samples is not None else info.source_rows

    if info.mode == "threshold":
        mode_description = f"threshold >= {info.threshold}"
    elif info.mode == "sample_without_replacement":
        mode_description = f"sample without replacement (n={n_display:,})"
    else:
        cap = (
            f", max_duplications={info.max_duplications}"
            if info.max_duplications is not None
            else ""
        )
        mode_description = f"sample with replacement (n={n_display:,}{cap})"

    content = _TEMPLATE.format(
        name=info.name,
        source_dataset=info.source_dataset,
        annotations_path=info.annotations_path,
        config_name=info.config_name,
        mode_description=mode_description,
        temperature=info.temperature,
        sampling=info.sampling,
        source_rows=info.source_rows,
        selected_rows=info.selected_rows,
        selection_pct=info.selected_rows / info.source_rows * 100,
        seed=info.seed,
        writer=info.writer,
        date=date.today().isoformat(),
        composition=_composition_section(info),
        resampling=_resampling_section(info),
    )

    readme_path = Path(output_dir) / "README.md"
    readme_path.write_text(content)
