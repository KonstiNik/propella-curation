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
filtered, scores = sampler.apply(ds, "annotations.parquet", mode="threshold", threshold=0.7)

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

`--dataset_path` accepts either a HuggingFace dataset directory (with a `data/` subdirectory containing parquets) or a direct path to a directory of parquet files.

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

The current implementation loads the full dataset into memory. Expect **~2x the uncompressed dataset size** in peak memory (check `num_bytes` in your dataset's `dataset_info`). Processing time scales linearly — ~2M rows takes ~2 minutes.

For SLURM clusters, a minimal submission looks like:

```bash
sbatch --wrap="propella-score-sampler --dataset_path ... --output_dir ..." \
  --partition=your_partition --account=your_account \
  --time=00:30:00 --mem=64G --cpus-per-task=4
```

> **Note:** For very large datasets (100GB+), the in-memory approach will not work. Streaming/iterative processing is planned for a future version.
