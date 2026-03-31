"""Generate a HuggingFace-compatible dataset card for curated datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

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
[propella-curation](https://github.com/OpenEuroLLM/propella-curation).

## Curation details

| Parameter | Value |
|-----------|-------|
| Source dataset | `{source_dataset}` |
| Annotations | `{annotations_path}` |
| Config | {config_name} |
| Mode | {mode_description} |
| Source rows | {source_rows:,} |
| Selected rows | {selected_rows:,} ({selection_pct:.1f}%) |
| Seed | {seed} |
| Date | {date} |

## Score distribution (after selection)

```
Mean={mean:.3f}  Median={median:.3f}  Std={std:.3f}
Min={min:.3f}  Max={max:.3f}
p10={p10:.3f}  p25={p25:.3f}  p50={p50:.3f}  p75={p75:.3f}  p90={p90:.3f}
```

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
    scores_after: "numpy.ndarray"  # noqa: F821


def write_dataset_card(info: CurationInfo, output_dir: str) -> None:
    """Write a README.md dataset card to the output directory."""
    import numpy as np

    scores = info.scores_after
    pcts = np.percentile(scores, [10, 25, 50, 75, 90])

    if info.mode == "threshold":
        mode_description = f"threshold >= {info.threshold}"
    elif info.mode == "sample_without_replacement":
        mode_description = f"sample without replacement (n={info.n_samples:,})"
    else:
        mode_description = f"sample with replacement (n={info.n_samples:,})"

    content = _TEMPLATE.format(
        name=info.name,
        source_dataset=info.source_dataset,
        annotations_path=info.annotations_path,
        config_name=info.config_name,
        mode_description=mode_description,
        source_rows=info.source_rows,
        selected_rows=info.selected_rows,
        selection_pct=info.selected_rows / info.source_rows * 100,
        seed=info.seed,
        date=date.today().isoformat(),
        mean=scores.mean(),
        median=np.median(scores),
        std=scores.std(),
        min=scores.min(),
        max=scores.max(),
        p10=pcts[0],
        p25=pcts[1],
        p50=pcts[2],
        p75=pcts[3],
        p90=pcts[4],
    )

    readme_path = Path(output_dir) / "README.md"
    readme_path.write_text(content)
