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
    filtered, scores, indices = sampler.apply(
        dataset, "annotations.parquet", mode="threshold", threshold=0.7
    )
    # `indices` are int64 source-table positions; for sample_with_replacement
    # they may contain duplicates.
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
import pyarrow as pa
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
        max_duplications: Optional[int] = None,
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
        nonzero = int((probs > 0).sum())

        if not replace:
            if n_samples > nonzero:
                n_samples = nonzero
                print(
                    f"  Warning: n_samples capped to {n_samples} "
                    f"(number of examples with score > 0) for sampling without replacement"
                )
            return rng.choice(len(scores), size=n_samples, replace=False, p=probs)

        # replace=True
        if max_duplications is None:
            return rng.choice(len(scores), size=n_samples, replace=True, p=probs)

        if max_duplications < 1:
            raise ValueError("max_duplications must be >= 1")

        capacity = max_duplications * nonzero
        if n_samples > capacity:
            print(
                f"  Warning: n_samples capped to {capacity} "
                f"(max_duplications={max_duplications} × {nonzero} eligible examples)"
            )
            n_samples = capacity

        # Iterative refill: draw with replacement, clamp any item that exceeds
        # the cap, then re-draw the excess from a renormalized distribution that
        # excludes saturated items. This produces the same distribution as
        # rejection sampling ("draw, redraw if the chosen item is already at
        # cap"): once an item is saturated, conditioning on it not being drawn
        # is equivalent to setting its probability to zero, and zeroing
        # preserves the relative weights of the remaining items.
        counts = np.zeros(len(scores), dtype=np.int64)
        remaining = n_samples
        cur_probs = probs.copy()

        while remaining > 0:
            draw = rng.choice(len(scores), size=remaining, replace=True, p=cur_probs)
            np.add.at(counts, draw, 1)
            over = counts > max_duplications
            if not over.any():
                break
            excess = int((counts[over] - max_duplications).sum())
            counts[over] = max_duplications
            cur_probs[counts >= max_duplications] = 0.0
            s = cur_probs.sum()
            if s == 0:
                # Unreachable: the capacity check above guarantees we never
                # exhaust the eligible pool before hitting n_samples.
                raise AssertionError(
                    "refill loop exhausted eligible pool — capacity check is broken"
                )
            cur_probs = cur_probs / s
            remaining = excess

        indices = np.repeat(np.arange(len(scores)), counts)
        rng.shuffle(indices)
        return indices

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
        max_duplications: Optional[int] = None,
    ) -> tuple[Dataset, np.ndarray, np.ndarray]:
        """Apply score-based selection.

        Returns ``(filtered_dataset, selected_scores, indices)`` where ``indices``
        is the int64 array of **source-table** row positions used to build the
        filtered dataset (may contain duplicates for ``sample_with_replacement``).
        If ``dataset`` has its own indices mapping (e.g. it was already
        ``select``-ed), those mappings are composed away — the returned indices
        always reference ``dataset.data.table`` directly, so a caller can do
        ``dataset.data.table.take(indices)`` and get the right rows.

        ``max_duplications`` is only meaningful with ``mode='sample_with_replacement'``;
        passing it with any other mode raises ValueError.
        """
        print("Score Sampler")
        print("=" * 50)

        if max_duplications is not None and mode != "sample_with_replacement":
            raise ValueError(
                f"max_duplications is only valid with mode='sample_with_replacement', "
                f"got mode='{mode}'"
            )

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
            indices = self._select_probabilistic(
                scores,
                n,
                replace=True,
                rng=rng,
                max_duplications=max_duplications,
            )
            cap_str = (
                f", max_dup={max_duplications}" if max_duplications is not None else ""
            )
            print(f"\n  Mode: sample with replacement (n={n:,}, seed={seed}{cap_str})")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(
            f"  Selected:           {len(indices):,} examples "
            f"({len(indices) / len(dataset) * 100:.1f}% of original)"
        )
        self._print_score_distribution(scores[indices], label="after selection")
        print("=" * 50)

        indices_i64 = np.asarray(indices, dtype=np.int64)
        # Compose any existing indices mapping into source-table coordinates so
        # the returned indices array is always directly take-able from
        # dataset.data.table — without this, the returned Dataset and the
        # returned indices reference different coordinate systems.
        if dataset._indices is not None:
            base = dataset._indices.column(0).to_numpy().astype(np.int64)
            source_indices = base[indices_i64]
        else:
            source_indices = indices_i64
        return dataset.select(indices_i64), scores[indices_i64], source_indices

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
# WRITER HELPERS
# ============================================================


def _load_chunks_in_memory(source_files: list[str]) -> list["pa.Table"]:
    """Fully decode each source parquet into an in-RAM single-chunk Arrow Table.

    Random access is then a pure memory operation, so the writer can preserve
    the sampler's draw order without per-shard random parquet decoding.
    """
    return [pq.read_table(p) for p in source_files]


def _load_chunks_mmap(dataset: Dataset) -> list["pa.Table"]:
    """Wrap each row-aligned batch of an mmap'd HF Dataset as its own Table.

    Uses ``Table.to_batches()`` which yields ``RecordBatch`` objects covering
    uniform row ranges across all columns, regardless of how the underlying
    columns are chunked. This sidesteps the need to validate chunk alignment
    and avoids ``combine_chunks`` (which would overflow 32-bit string offsets
    on large nested columns).

    No copies — the resulting tables share the underlying mmap'd buffers.
    """
    return [pa.Table.from_batches([batch]) for batch in dataset.data.table.to_batches()]


def _estimate_decoded_size_bytes(source_files: list[str]) -> int:
    """Cheap estimate of decoded source size by reading parquet metadata only.

    Sums each file's uncompressed total byte size from row-group metadata.
    No decoding, so this is fast (~ms per file). The estimate is approximate
    because parquet's recorded uncompressed size doesn't account for Arrow's
    in-memory layout overhead, but it's good enough for a soft warning.
    """
    total = 0
    for p in source_files:
        meta = pq.ParquetFile(p).metadata
        for rg in range(meta.num_row_groups):
            total += meta.row_group(rg).total_byte_size
    return total


def _available_memory_bytes() -> Optional[int]:
    """Best-effort detection of available RAM, preferring SLURM cgroup limit.

    Returns None if we can't tell. SLURM_MEM_PER_NODE is documented as MB but
    may carry a unit suffix (K/M/G/T) depending on cluster config.
    """
    slurm_mem = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mem:
        try:
            s = slurm_mem.strip().upper()
            unit = 1024 * 1024  # default: MB
            if s.endswith(("K", "KB")):
                s, unit = s.rstrip("KB"), 1024
            elif s.endswith(("M", "MB")):
                s, unit = s.rstrip("MB"), 1024 * 1024
            elif s.endswith(("G", "GB")):
                s, unit = s.rstrip("GB"), 1024 * 1024 * 1024
            elif s.endswith(("T", "TB")):
                s, unit = s.rstrip("TB"), 1024 * 1024 * 1024 * 1024
            return int(float(s) * unit)
        except (ValueError, TypeError):
            pass
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().available
    except Exception:
        return None


def _write_sharded_parquet(
    src_chunks: list["pa.Table"],
    gather_indices: np.ndarray,
    rows_per_shard: int,
    output_dir: str,
    *,
    sort_for_locality: bool,
) -> int:
    """Write a sharded parquet output by gathering rows from per-chunk tables.

    Each gather index points into the logical concatenation of ``src_chunks``.
    Per shard we group indices by source chunk and call ``Table.take`` once
    per chunk, then concat the small per-chunk results. This avoids both
    HF's row-by-row materialization and pyarrow's chunked-array concat (which
    overflows 32-bit string offsets on large nested columns).

    If ``sort_for_locality`` is True, gather_indices is sorted globally so
    each shard touches contiguous source rows — required for the streaming
    (mmap'd) writer to avoid decoding parquet pages randomly. Otherwise the
    indices are written in the order ScoreSampler produced them (which for
    sample_with_replacement is itself the post-shuffle output of the refill
    loop, not literal rng.choice draw order).
    """
    if sort_for_locality:
        gather_indices = np.sort(gather_indices)
    else:
        gather_indices = np.asarray(gather_indices, dtype=np.int64)

    chunk_lens = [len(c) for c in src_chunks]
    total_rows = int(np.sum(chunk_lens))
    if len(gather_indices) and (
        gather_indices.min() < 0 or gather_indices.max() >= total_rows
    ):
        raise IndexError(
            f"gather index out of range: min={gather_indices.min()}, "
            f"max={gather_indices.max()}, source rows={total_rows}"
        )
    chunk_offsets = np.concatenate([[0], np.cumsum(chunk_lens)]).astype(np.int64)
    chunk_ids_all = np.searchsorted(chunk_offsets[1:], gather_indices, side="right")
    local_idx_all = gather_indices - chunk_offsets[chunk_ids_all]

    n_total = len(gather_indices)
    n_shards = max(1, (n_total + rows_per_shard - 1) // rows_per_shard)

    # Empty input: still write a single empty shard with the source schema so
    # downstream consumers see a valid (empty) HF dataset layout.
    if n_total == 0:
        empty = src_chunks[0].slice(0, 0) if src_chunks else None
        if empty is not None:
            shard_path = os.path.join(output_dir, "train-00000-of-00001.parquet")
            pq.write_table(empty, shard_path, use_dictionary=True, compression="snappy")
            print(f"  wrote shard 1/1 (0 rows)")
        return 1

    for i in range(n_shards):
        start = i * rows_per_shard
        end = min((i + 1) * rows_per_shard, n_total)
        shard_chunk_ids = chunk_ids_all[start:end]
        shard_local = local_idx_all[start:end]

        # Group rows by source chunk so each take() hits a single chunk (avoids
        # the chunked-array concat that overflows 32-bit string offsets), then
        # restore the original within-shard order via an inverse permutation.
        # When sort_for_locality=True the indices were globally sorted upstream,
        # so chunk grouping is already in order and the inverse permutation is
        # the identity — but it's cheap and keeps both paths uniform.
        grouped_order = np.argsort(shard_chunk_ids, kind="stable")
        grouped_chunk_ids = shard_chunk_ids[grouped_order]
        grouped_local = shard_local[grouped_order]

        parts = []
        run_starts = np.flatnonzero(
            np.r_[True, grouped_chunk_ids[1:] != grouped_chunk_ids[:-1]]
        )
        run_ends = np.r_[run_starts[1:], len(grouped_chunk_ids)]
        for run_start, run_end in zip(run_starts, run_ends):
            ci = int(grouped_chunk_ids[run_start])
            parts.append(
                src_chunks[ci].take(pa.array(grouped_local[run_start:run_end]))
            )
        shard_grouped = pa.concat_tables(parts) if len(parts) > 1 else parts[0]

        if sort_for_locality:
            # grouped_order is monotonic in this case; skip the inverse take.
            shard = shard_grouped
        else:
            inverse_order = np.empty(len(grouped_order), dtype=np.int64)
            inverse_order[grouped_order] = np.arange(len(grouped_order), dtype=np.int64)
            shard = shard_grouped.take(pa.array(inverse_order))

        shard_path = os.path.join(
            output_dir, f"train-{i:05d}-of-{n_shards:05d}.parquet"
        )
        pq.write_table(shard, shard_path, use_dictionary=True, compression="snappy")
        print(f"  wrote shard {i + 1}/{n_shards} ({len(shard):,} rows)")
    return n_shards


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
        "--max_duplications",
        type=int,
        default=None,
        help="Cap on how many times any single example may be drawn "
        "(only used with --mode sample_with_replacement). Default: no cap. "
        "If --n_samples exceeds max_duplications × (number of examples with score > 0), "
        "n_samples is capped to that capacity and a warning is printed.",
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
    optional.add_argument(
        "--writer",
        choices=["load", "stream"],
        default="load",
        help="How to access the source dataset when writing shards. "
        "'load' (default): fully decode every source parquet into RAM, then "
        "write shards in the sampler's draw order. Fast and order-preserving, "
        "but peak RAM ≈ decoded source size. "
        "'stream': access source parquets via mmap and gather chunkwise. Peak "
        "RAM ≈ one shard, but output rows are reordered (sorted by source row "
        "position) — use this for sources that don't fit in node RAM.",
    )

    args = parser.parse_args()

    print("propella-score-sampler")
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

    # Pick a per-shard row count that mirrors the source layout (same shard
    # count as input when not over- or undersampling). Floor to 1 so tiny
    # outputs (e.g. heavy threshold filtering) don't divide to 0.
    n_source_files = max(1, len(source_files)) if source_files else 1
    rows_per_shard = max(1, len(ds) // n_source_files)

    sampler = ScoreSampler(config=config)
    filtered, scores_after, gather_indices = sampler.apply(
        ds,
        annotations_path=args.annotations_path,
        mode=args.mode,
        threshold=args.threshold,
        n_samples=args.n_samples,
        seed=args.seed,
        force=args.force,
        max_duplications=args.max_duplications,
    )

    # `gather_indices` from apply() is already in source-table coordinates.
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Load source chunks according to the chosen writer mode.
    if args.writer == "load":
        if source_files:
            est = _estimate_decoded_size_bytes(source_files)
            avail = _available_memory_bytes()
            if avail is not None and est > 0.8 * avail:
                print(
                    f"\n  Warning: estimated decoded source size "
                    f"~{est / 1e9:.1f} GB exceeds 80% of available memory "
                    f"(~{avail / 1e9:.1f} GB). The 'load' writer may OOM. "
                    f"Consider --writer stream (lower RAM, but output rows "
                    f"are reordered)."
                )
            print(f"\nLoading {len(source_files)} source parquet(s) into memory ...")
            try:
                src_chunks = _load_chunks_in_memory(source_files)
            except MemoryError as e:
                raise MemoryError(
                    f"Failed to load source dataset into memory "
                    f"(estimated decoded size ~{est / 1e9:.1f} GB).\n"
                    f"Re-run with --writer stream to process the dataset "
                    f"chunkwise:\n"
                    f"  - Lower peak memory (~one shard at a time)\n"
                    f"  - Output rows will be reordered (sorted by source row "
                    f"position) rather than preserved in sampler draw order."
                ) from e
        else:
            # HF-hub-loaded datasets don't expose stable source files we can
            # re-decode with pq.read_table. Falling through to mmap chunks
            # would silently change semantics (no order preservation, no
            # eager load), so we error out instead.
            raise NotImplementedError(
                "--writer load is only supported for local parquet datasets. "
                "Re-run with --writer stream, or download the dataset to a "
                "local parquet directory and pass that as --dataset_path."
            )
        sort_for_locality = False
    else:  # args.writer == "stream"
        src_chunks = _load_chunks_mmap(ds)
        sort_for_locality = True

    n_shards = _write_sharded_parquet(
        src_chunks,
        gather_indices,
        rows_per_shard,
        data_dir,
        sort_for_locality=sort_for_locality,
    )

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
        max_duplications=args.max_duplications,
        writer=args.writer,
    )
    write_dataset_card(card_info, args.output_dir)
    print(f"  Dataset card written to {args.output_dir}/README.md")


if __name__ == "__main__":
    main()
