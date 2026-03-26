# propella-curation

Filters and resamples HuggingFace datasets based on quality scores from [propella annotations](https://github.com/OpenEuroLLM/propella-annotations/tree/main). Produces a new dataset ready for tokenization — agnostic to the training framework.

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
filtered = sampler.apply(ds, "annotations.parquet", mode="threshold", threshold=0.7)

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

Use `--config propella_all` for all 7 annotation columns, or `--config /path/to/custom.yaml` for a custom config. See `propella-score-sampler --help` for all options.
