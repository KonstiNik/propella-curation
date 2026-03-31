"""Score-based data sampling for SFT quality filtering.

Loads a propella-annotations parquet, computes per-example quality scores
from categorical labels, and selects examples via one of three strategies:

  - threshold:                     keep examples with score >= threshold
  - sample_without_replacement:    draw N examples, P(select) proportional to score
  - sample_with_replacement:       same, but examples can repeat (oversample high-quality)

Usage (CLI):
    propella-score-sampler \
        --dataset_path /path/to/dataset \
        --annotations_path /path/to/annotations.parquet \
        --output_dir /path/to/output \
        --mode threshold --threshold 0.7 --seed 42

Usage (library):
    from propella_curation import ScoreSampler
    sampler = ScoreSampler()
    filtered, scores = sampler.apply(dataset, "annotations.parquet", mode="threshold", threshold=0.7)
"""

from __future__ import annotations

import argparse
import glob as globmod
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


# ============================================================
# SCORING CONFIGURATION
# ============================================================

_CONFIGS_DIR = Path(__file__).parent / "configs"


@dataclass
class ColumnScoring:
    """Scoring rule for a single annotation column."""

    category_scores: Dict[str, float]  # category value -> numeric score
    weight: float = 1.0  # relative importance in the composite score
    default_score: float = 0.5  # fallback for categories not listed above


@dataclass
class ScoringConfig:
    """Complete scoring configuration."""

    columns: Dict[str, ColumnScoring]
    aggregation: Literal["weighted_mean", "weighted_sum", "min", "product"] = (
        "weighted_mean"
    )
    missing_id_score: float = 0.0  # score assigned when an ID has no annotation
    normalize: bool = True  # min-max rescale final scores to [0, 1]

    @classmethod
    def from_file(cls, path: str | Path) -> ScoringConfig:
        """Load a scoring config from a YAML file."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def from_name(cls, name: str) -> ScoringConfig:
        """Load a bundled scoring config by name (without .yaml extension)."""
        path = _CONFIGS_DIR / f"{name}.yaml"
        if not path.exists():
            available = [p.stem for p in _CONFIGS_DIR.glob("*.yaml")]
            raise FileNotFoundError(
                f"No bundled config named '{name}'. Available: {available}"
            )
        return cls.from_file(path)

    @classmethod
    def _from_dict(cls, raw: dict) -> ScoringConfig:
        columns = {}
        for col_name, col_data in raw.get("columns", {}).items():
            columns[col_name] = ColumnScoring(
                category_scores=col_data["category_scores"],
                weight=col_data.get("weight", 1.0),
                default_score=col_data.get("default_score", 0.5),
            )
        return cls(
            columns=columns,
            aggregation=raw.get("aggregation", "weighted_mean"),
            missing_id_score=raw.get("missing_id_score", 0.0),
            normalize=raw.get("normalize", True),
        )


DEFAULT_SCORING_CONFIG = ScoringConfig.from_name("default")


# ============================================================
# SAMPLER
# ============================================================


class ScoreSampler:
    """Score-based dataset sampler."""

    def __init__(self, config: ScoringConfig = DEFAULT_SCORING_CONFIG):
        self.config = config

    # ------------------------------------------------------------------
    # Loading & scoring
    # ------------------------------------------------------------------

    def load_annotations(self, path: str) -> pd.DataFrame:
        """Load annotations parquet (file or directory) into a DataFrame.

        Only reads the ``id`` column plus columns referenced in the config.
        """
        needed_cols = ["id"] + list(self.config.columns.keys())
        df = pd.read_parquet(path, columns=needed_cols)
        return df

    def compute_scores(self, annotations_df: pd.DataFrame) -> pd.Series:
        """Return an id-indexed Series of float quality scores."""
        df = annotations_df.set_index("id")
        col_scores: list[pd.Series] = []
        weights: list[float] = []

        for col_name, col_cfg in self.config.columns.items():
            if col_name not in df.columns:
                continue
            mapped = (
                df[col_name].map(col_cfg.category_scores).fillna(col_cfg.default_score)
            )
            col_scores.append(mapped)
            weights.append(col_cfg.weight)

        if not col_scores:
            raise ValueError("No annotation columns matched the scoring config")

        scores_matrix = pd.concat(col_scores, axis=1)
        weight_arr = np.array(weights)

        agg = self.config.aggregation
        if agg == "weighted_mean":
            raw = (scores_matrix.values * weight_arr).sum(axis=1) / weight_arr.sum()
        elif agg == "weighted_sum":
            raw = (scores_matrix.values * weight_arr).sum(axis=1)
        elif agg == "min":
            raw = scores_matrix.values.min(axis=1)
        elif agg == "product":
            raw = scores_matrix.values.prod(axis=1)
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

        result = pd.Series(raw, index=df.index, name="score")

        if self.config.normalize:
            lo, hi = result.min(), result.max()
            if hi > lo:
                result = (result - lo) / (hi - lo)

        return result

    # ------------------------------------------------------------------
    # Selection strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _select_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
        return np.where(scores >= threshold)[0]

    @staticmethod
    def _select_probabilistic(
        scores: np.ndarray,
        n_samples: int,
        replace: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        probs = scores.copy()
        probs[probs < 0] = 0.0
        total = probs.sum()
        if total == 0:
            raise ValueError(
                "All scores are zero — cannot sample. "
                "Lower the threshold or adjust the scoring config."
            )
        probs = probs / total
        if not replace and n_samples > (probs > 0).sum():
            n_samples = int((probs > 0).sum())
            print(
                f"  Warning: n_samples capped to {n_samples} "
                f"(number of examples with score > 0) for sampling without replacement"
            )
        return rng.choice(len(scores), size=n_samples, replace=replace, p=probs)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def apply(
        self,
        dataset: Dataset,
        annotations_path: str,
        mode: Literal[
            "threshold", "sample_without_replacement", "sample_with_replacement"
        ],
        threshold: float = 0.5,
        n_samples: Optional[int] = None,
        seed: int = 42,
        force: bool = False,
    ) -> tuple[Dataset, np.ndarray]:
        """Apply score-based selection and return (filtered Dataset, selected scores)."""
        print("Score Sampler")
        print("=" * 50)

        # 1. Load annotations & compute scores
        ann_df = self.load_annotations(annotations_path)
        score_series = self.compute_scores(ann_df)
        print(f"  Annotations loaded: {len(ann_df):,}")

        # 2. Map dataset ids to scores (preserving dataset order)
        dataset_ids = dataset["id"]
        id_to_score = score_series.to_dict()
        scores = np.array(
            [id_to_score.get(did, self.config.missing_id_score) for did in dataset_ids]
        )
        n_matched = sum(1 for did in dataset_ids if did in id_to_score)
        match_rate = n_matched / len(dataset_ids) if dataset_ids else 0.0
        print(f"  Dataset size:       {len(dataset_ids):,}")
        print(f"  IDs matched:        {n_matched:,} ({match_rate * 100:.1f}%)")

        if match_rate < 0.2:
            msg = (
                f"Only {match_rate * 100:.1f}% of dataset IDs have annotations. "
                f"Check that the annotation file matches the dataset."
            )
            if force:
                logger.warning(msg + " Continuing because force=True.")
            else:
                raise ValueError(msg)
        elif match_rate < 0.9:
            logger.warning(
                f"{match_rate * 100:.1f}% of dataset IDs have annotations — "
                f"{len(dataset_ids) - n_matched:,} examples will receive a score of "
                f"{self.config.missing_id_score}."
            )

        self._print_score_distribution(scores, label="before selection")

        # 3. Select indices
        if mode == "threshold":
            indices = self._select_threshold(scores, threshold)
            print(f"\n  Mode: threshold (>= {threshold})")
        elif mode == "sample_without_replacement":
            n = n_samples if n_samples is not None else len(dataset)
            rng = np.random.default_rng(seed)
            indices = self._select_probabilistic(scores, n, replace=False, rng=rng)
            print(f"\n  Mode: sample without replacement (n={n:,}, seed={seed})")
        elif mode == "sample_with_replacement":
            n = n_samples if n_samples is not None else len(dataset)
            rng = np.random.default_rng(seed)
            indices = self._select_probabilistic(scores, n, replace=True, rng=rng)
            print(f"\n  Mode: sample with replacement (n={n:,}, seed={seed})")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(
            f"  Selected:           {len(indices):,} examples "
            f"({len(indices) / len(dataset) * 100:.1f}% of original)"
        )
        self._print_score_distribution(scores[indices], label="after selection")
        print("=" * 50)

        return dataset.select(indices), scores[indices]

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _print_score_distribution(scores: np.ndarray, label: str) -> None:
        print(f"\n  Score distribution ({label}):")
        print(
            f"    Mean={scores.mean():.3f}  Median={np.median(scores):.3f}  "
            f"Std={scores.std():.3f}"
        )
        print(f"    Min={scores.min():.3f}  Max={scores.max():.3f}")
        pcts = np.percentile(scores, [10, 25, 50, 75, 90])
        print(
            f"    p10={pcts[0]:.3f}  p25={pcts[1]:.3f}  p50={pcts[2]:.3f}  "
            f"p75={pcts[3]:.3f}  p90={pcts[4]:.3f}"
        )


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score-based sampling for SFT datasets using propella annotations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--dataset_path",
        required=True,
        help="Path to local dataset directory (looks for parquets in data/ subdirectory, "
        "then falls back to the given path) or HuggingFace dataset name",
    )
    required.add_argument(
        "--annotations_path",
        required=True,
        help="Path to propella-annotations parquet file or directory",
    )
    required.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the filtered dataset (parquet shards)",
    )
    required.add_argument(
        "--mode",
        required=True,
        choices=["threshold", "sample_without_replacement", "sample_with_replacement"],
        help="Selection strategy. 'threshold' keeps all examples with score >= --threshold. "
        "'sample_without_replacement' draws --n_samples examples weighted by score (no repeats). "
        "'sample_with_replacement' allows repeats, enabling oversampling of high-quality examples.",
    )

    optional = parser.add_argument_group("optional arguments")
    optional.add_argument(
        "--dataset_split",
        default="train",
        help="Dataset split to load (default: train)",
    )
    optional.add_argument(
        "--config",
        default=None,
        help="Scoring config: a bundled name (e.g. 'propella_all') or path to a YAML file. "
        "Defaults to 'default' (content_quality only).",
    )
    optional.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum score to keep an example (only used with --mode threshold). Default: 0.5.",
    )
    optional.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of examples to draw (only used with sampling modes). "
        "Defaults to dataset size. Set higher than dataset size with "
        "'sample_with_replacement' to oversample.",
    )
    optional.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling. Default: 42.",
    )
    optional.add_argument(
        "--force",
        action="store_true",
        help="Continue even if less than 20%% of dataset IDs have annotations.",
    )

    args = parser.parse_args()

    print(f"propella-score-sampler")
    print(f"  Dataset:     {args.dataset_path}")
    print(f"  Annotations: {args.annotations_path}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Mode:        {args.mode}")
    print(f"  Config:      {args.config or 'default'}")

    # Load scoring config
    if args.config is None:
        config = DEFAULT_SCORING_CONFIG
    elif os.path.isfile(args.config):
        config = ScoringConfig.from_file(args.config)
    else:
        config = ScoringConfig.from_name(args.config)

    # Load dataset — local parquet directory or HF name
    print(f"\nLoading dataset from {args.dataset_path} ...")
    if os.path.exists(args.dataset_path):
        data_subdir = os.path.join(args.dataset_path, "data")
        if os.path.isdir(data_subdir) and globmod.glob(f"{data_subdir}/*.parquet"):
            parquet_dir = data_subdir
        else:
            parquet_dir = args.dataset_path
        source_files = sorted(globmod.glob(f"{parquet_dir}/*.parquet"))
        if not source_files:
            raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")
        ds = load_dataset("parquet", data_files=source_files, split=args.dataset_split)
    else:
        source_files = []
        ds = load_dataset(args.dataset_path, split=args.dataset_split)
    print(f"Loaded dataset: {len(ds):,} rows, columns: {ds.column_names}")

    # Estimate rows per shard from source parquet files
    if source_files:
        rows_per_shard = len(ds) // len(source_files)
    else:
        rows_per_shard = len(ds)

    sampler = ScoreSampler(config=config)
    filtered, scores_after = sampler.apply(
        ds,
        annotations_path=args.annotations_path,
        mode=args.mode,
        threshold=args.threshold,
        n_samples=args.n_samples,
        seed=args.seed,
        force=args.force,
    )

    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    table = filtered.data.table
    n_shards = max(1, (len(filtered) + rows_per_shard - 1) // rows_per_shard)
    for i in range(n_shards):
        start = i * rows_per_shard
        end = min((i + 1) * rows_per_shard, len(filtered))
        shard = table.slice(start, end - start)
        shard_path = os.path.join(data_dir, f"train-{i:05d}-of-{n_shards:05d}.parquet")
        pq.write_table(shard, shard_path, use_dictionary=True, compression="snappy")

    ratio = len(filtered) / len(ds)
    print(f"\nSaved {len(filtered):,} examples to {data_dir}/ ({n_shards} shards)")
    print(f"  Selection ratio: {ratio:.2f}x ({ratio * 100:.1f}% of original)")

    # Write dataset card
    from propella_curation.dataset_card import CurationInfo, write_dataset_card

    card_info = CurationInfo(
        name=Path(args.output_dir).name,
        source_dataset=args.dataset_path,
        annotations_path=args.annotations_path,
        config_name=args.config or "default",
        mode=args.mode,
        threshold=args.threshold,
        n_samples=args.n_samples,
        seed=args.seed,
        source_rows=len(ds),
        selected_rows=len(filtered),
        scores_after=scores_after,
    )
    write_dataset_card(card_info, args.output_dir)
    print(f"  Dataset card written to {args.output_dir}/README.md")


if __name__ == "__main__":
    main()
