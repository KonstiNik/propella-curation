# propella-curation — agent notes

Score-based resampling of SFT datasets using propella annotations. Maps
annotation labels → a composite quality score, then selects/resamples
(threshold, with/without replacement, temperature tilt). See `docs/math.md` for
the full math and the design rationale behind the curation sweeps.

## Common commands

```bash
source .venv/bin/activate                      # or prefix with .venv/bin/ or `uv run`
python -m pytest -q                            # tests
propella-gen-sweep sweeps/<spec>.yaml          # generate a sweep (configs + manifest + submit_all.sh)
bash sweeps/<name>.submit_all.sh               # launch one SLURM job per temperature
propella-score-sampler --help                  # single curation run
```

## Dataset card contents

The curated dataset card (`dataset_card.py`) reports two things, because both
matter when comparing curated SFT datasets:

- **(a) Quality composition, before → after** — per-value counts/shares of the
  composite score. Shown as a labelled table when the score is **discrete**
  (≤ 20 distinct values, e.g. single-column `content_quality`); tier names are
  used when the mapping is unambiguous (single column + `normalize: false`).
- **(b) Coverage / resampling profile** — unique source rows kept, dropped, max
  copies of any row, and a copies histogram. Computed from the gather indices;
  meaningful for every mode (degenerates to all-unique for `threshold`).

**Deferred (value-adaptive continuous branch):** for a *continuous* composite
(many distinct values, e.g. `propella_all` with `weighted_mean`), the card
currently falls back to the percentile summary (mean/median/std/min/max +
p10..p90), which is appropriate there. A proper **binned histogram** for that
branch is a nice-to-have, not yet implemented. The discrete→table /
continuous→summary switch is the "value-adaptive" mode.

## Tech debt / cleanup notes

### `ScoreSampler.apply()` returns the selection twice in two coordinate systems

[`score_sampler.py:401`](src/propella_curation/score_sampler.py#L401) returns
`(dataset.select(indices), scores[indices], source_indices)`. The first and
third elements both encode *which rows were selected*, but in different
coordinate spaces — HF's internal `_indices` mapping vs. raw source-table
positions — reconciled by the composition block at
[`score_sampler.py:393-401`](src/propella_curation/score_sampler.py#L393-L401)
(and the docstring at
[`score_sampler.py:295-301`](src/propella_curation/score_sampler.py#L295-L301)).
The `Dataset` view is only ever used for `len(filtered)`.

**Why it matters (latent, not an active bug):** the two look interchangeable but
only `source_indices` is safe to write from. A plausible future "simplification"
(write `filtered` directly instead of gathering from source parquet via
`source_indices`) would **silently write the wrong rows** when the input dataset
was already `.select`-ed, and could break the `writer: load` draw-order
guarantee that the order analysis depends on — both with no loud test failure.

**Suggested fix:** return a small named result instead of a positional triple —
`SelectionResult(source_indices, scores)` in one coordinate system — and drop the
`Dataset` return (callers use `len(source_indices)`). Contained change: one
return type, ~3 call sites, the card, a few tests. Do it on a **separate
branch/PR**, not the curation-sweep branch.

History: `apply()` originally returned a single `dataset.select(indices)`
(initial commit `6bed18a`); the triple was added with the writer-modes PR
(`0634813`) without introducing a result type.
