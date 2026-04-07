# propella-curation

Filters and resamples HuggingFace datasets based on quality scores from [propella annotations](https://github.com/OpenEuroLLM/propella-annotations/tree/main). Produces a new dataset ready for tokenization. 

Each propella annotation has 17 categorical columns (e.g. `content_quality: "excellent"`, `content_safety: "safe"`). A YAML scoring config maps selected columns to numeric scores, assigns weights, and aggregates them into a single composite score per example. Examples are then kept by threshold or sampled proportionally to their score. See [`src/propella_curation/configs/`](src/propella_curation/configs/) for bundled configs.

## Prerequisites

- A HuggingFace dataset with an `id` column
- A propella annotations parquet with matching IDs

## Installation

```bash
pip install -e .
```

## Usage

### Python

```python
from propella_curation import ScoreSampler, ScoringConfig
from datasets import load_dataset

ds = load_dataset("parquet", data_files="data/*.parquet", split="train")

sampler = ScoreSampler()  # default: scores on content_quality only
filtered, scores, indices = sampler.apply(
    ds, "annotations.parquet", mode="threshold", threshold=0.7
)
# `indices` are int64 source-table positions; for sample_with_replacement
# they may contain duplicates.

# use a different bundled config
sampler = ScoreSampler(config=ScoringConfig.from_name("propella_all"))

# or a custom YAML config
sampler = ScoreSampler(config=ScoringConfig.from_file("my_config.yaml"))
```

### CLI

```bash
propella-score-sampler \
  --dataset_path /path/to/dataset \
  --annotations_path /path/to/annotations.parquet \
  --output_dir /path/to/output \
  --mode threshold --threshold 0.7
```

To **oversample** high-quality examples (draw with replacement, weighted by score), use `sample_with_replacement` and set `--n_samples` larger than the source size. Optionally cap how many times any single example may be duplicated with `--max_duplications`:

```bash
propella-score-sampler \
  --dataset_path /path/to/dataset \
  --annotations_path /path/to/annotations.parquet \
  --output_dir /path/to/output \
  --mode sample_with_replacement \
  --n_samples 4000000 \
  --max_duplications 5
```

If `--n_samples` exceeds `--max_duplications × (number of examples with score > 0)`, it is capped to that capacity.

`--dataset_path` accepts a local HuggingFace dataset directory (with a `data/` subdirectory containing parquets) or a direct path to a directory of parquet files. HF Hub dataset names are accepted only with `--writer stream`; the default `--writer load` requires local parquet files it can re-decode.

Use `--config propella_all` for all 7 annotation columns, or `--config /path/to/custom.yaml` for a custom config. See `propella-score-sampler --help` for all options.

### Output format

The output directory follows the HuggingFace dataset layout:

```
output_dir/
├── README.md                          # Dataset card with curation provenance
├── data/
│   ├── train-00000-of-00005.parquet
│   ├── train-00001-of-00005.parquet
│   └── ...
```

Shard sizes are matched to the source dataset. The dataset card records the source dataset, config, mode, threshold, selection ratio, and score distribution.

### Resource requirements

By default (`--writer load`) the source dataset is fully decoded into memory before writing shards. Expect peak RAM ≈ decoded source size + one shard (typically ~3–5× the compressed parquet size). The output preserves the sampler's draw order. Processing time scales linearly — ~2M rows takes ~2 minutes.

For sources that don't fit in node RAM, pass `--writer stream`. This accesses source parquets via mmap and gathers one shard at a time, keeping peak RAM at ~one shard. **Output rows are reordered** (sorted by source row position) rather than preserved in sampler draw order. This is invisible to most training pipelines because dataloaders shuffle every epoch, but if you depend on the exact draw order downstream, use the default.

For SLURM clusters, a minimal submission looks like:

```bash
sbatch --wrap="propella-score-sampler --dataset_path ... --output_dir ..." \
  --partition=your_partition --account=your_account \
  --time=00:30:00 --mem=64G --cpus-per-task=4
```

> **Note:** For very large datasets that don't fit in node RAM, use `--writer stream` (see Resource requirements above).
