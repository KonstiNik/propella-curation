"""Unit and integration tests for ScoreSampler."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from datasets import Dataset

from propella_curation import (
    DEFAULT_SCORING_CONFIG,
    ColumnScoring,
    ScoreSampler,
    ScoringConfig,
)
from propella_curation.labels import ORDINAL_LABELS

# ============================================================
# Fixtures — synthetic data matching real schema
# ============================================================


def _make_dataset(n: int = 100, id_prefix: str = "doc") -> Dataset:
    """Create a synthetic HF dataset with id and messages columns."""
    return Dataset.from_dict(
        {
            "id": [f"{id_prefix}_{i}" for i in range(n)],
            "messages": [
                [
                    {"role": "user", "content": f"question {i}"},
                    {"role": "assistant", "content": f"answer {i}"},
                ]
                for i in range(n)
            ],
            "source_dataset": ["test_source"] * n,
            "domain": ["test_domain"] * n,
        }
    )


_COL_PARAM_MAP = {
    "quality": "content_quality",
    "safety": "content_safety",
    "integrity": "content_integrity",
    "density": "information_density",
    "educational": "educational_value",
    "reasoning": "reasoning_indicators",
    "commercial": "commercial_bias",
}


def _make_annotations_df(
    ids: list[str],
    quality: list[str] | None = None,
    safety: list[str] | None = None,
    integrity: list[str] | None = None,
    density: list[str] | None = None,
    educational: list[str] | None = None,
    reasoning: list[str] | None = None,
    commercial: list[str] | None = None,
) -> pd.DataFrame:
    """Create a synthetic annotations DataFrame."""
    n = len(ids)
    rng = np.random.default_rng(0)
    overrides = {
        "quality": quality,
        "safety": safety,
        "integrity": integrity,
        "density": density,
        "educational": educational,
        "reasoning": reasoning,
        "commercial": commercial,
    }
    data: dict[str, list[str]] = {"id": ids}
    for param, col_name in _COL_PARAM_MAP.items():
        value = overrides[param]
        if value is not None:
            data[col_name] = value
        elif param == "quality":
            # Always include content_quality by default
            data[col_name] = rng.choice(ORDINAL_LABELS[col_name], n).tolist()
    return pd.DataFrame(data)


def _write_annotations_parquet(df: pd.DataFrame, tmpdir: str) -> str:
    """Write annotations DataFrame to a parquet file and return the path."""
    path = os.path.join(tmpdir, "annotations.parquet")
    df.to_parquet(path, index=False)
    return path


def _simple_config(weight: float = 1.0) -> ScoringConfig:
    """Config that only scores content_quality."""
    return ScoringConfig(
        columns={
            "content_quality": ColumnScoring(
                category_scores={
                    "excellent": 1.0,
                    "good": 0.75,
                    "adequate": 0.4,
                    "poor": 0.1,
                    "unacceptable": 0.0,
                },
                weight=weight,
            ),
        },
        aggregation="weighted_mean",
        normalize=False,
    )


# ============================================================
# Unit tests — compute_scores
# ============================================================


class TestComputeScores:
    def test_maps_categories_correctly(self):
        config = _simple_config()
        sampler = ScoreSampler(config=config)
        df = _make_annotations_df(
            ids=["a", "b", "c", "d", "e"],
            quality=["excellent", "good", "adequate", "poor", "unacceptable"],
        )
        scores = sampler.compute_scores(df)
        expected = [1.0, 0.75, 0.4, 0.1, 0.0]
        np.testing.assert_allclose(scores.values, expected)

    def test_default_score_for_unknown_category(self):
        config = _simple_config()
        sampler = ScoreSampler(config=config)
        df = _make_annotations_df(
            ids=["a", "b"],
            quality=["excellent", "never_seen_before"],
        )
        scores = sampler.compute_scores(df)
        assert scores["a"] == 1.0
        assert scores["b"] == config.columns["content_quality"].default_score

    def test_weighted_mean_aggregation(self):
        config = ScoringConfig(
            columns={
                "content_quality": ColumnScoring(
                    category_scores={"good": 0.8},
                    weight=2.0,
                ),
                "content_safety": ColumnScoring(
                    category_scores={"safe": 1.0},
                    weight=3.0,
                ),
            },
            aggregation="weighted_mean",
            normalize=False,
        )
        sampler = ScoreSampler(config=config)
        df = pd.DataFrame(
            {
                "id": ["a"],
                "content_quality": ["good"],
                "content_safety": ["safe"],
            }
        )
        scores = sampler.compute_scores(df)
        expected = (0.8 * 2.0 + 1.0 * 3.0) / (2.0 + 3.0)
        np.testing.assert_allclose(scores.values, [expected])

    def test_weighted_sum_aggregation(self):
        config = ScoringConfig(
            columns={
                "content_quality": ColumnScoring(
                    category_scores={"good": 0.8},
                    weight=2.0,
                ),
                "content_safety": ColumnScoring(
                    category_scores={"safe": 1.0},
                    weight=3.0,
                ),
            },
            aggregation="weighted_sum",
            normalize=False,
        )
        sampler = ScoreSampler(config=config)
        df = pd.DataFrame(
            {
                "id": ["a"],
                "content_quality": ["good"],
                "content_safety": ["safe"],
            }
        )
        scores = sampler.compute_scores(df)
        expected = 0.8 * 2.0 + 1.0 * 3.0
        np.testing.assert_allclose(scores.values, [expected])

    def test_min_aggregation(self):
        config = ScoringConfig(
            columns={
                "content_quality": ColumnScoring(
                    category_scores={"good": 0.8},
                    weight=2.0,
                ),
                "content_safety": ColumnScoring(
                    category_scores={"safe": 1.0},
                    weight=3.0,
                ),
            },
            aggregation="min",
            normalize=False,
        )
        sampler = ScoreSampler(config=config)
        df = pd.DataFrame(
            {
                "id": ["a"],
                "content_quality": ["good"],
                "content_safety": ["safe"],
            }
        )
        scores = sampler.compute_scores(df)
        np.testing.assert_allclose(scores.values, [0.8])

    def test_product_aggregation(self):
        config = ScoringConfig(
            columns={
                "content_quality": ColumnScoring(
                    category_scores={"good": 0.8},
                    weight=2.0,
                ),
                "content_safety": ColumnScoring(
                    category_scores={"safe": 0.5},
                    weight=3.0,
                ),
            },
            aggregation="product",
            normalize=False,
        )
        sampler = ScoreSampler(config=config)
        df = pd.DataFrame(
            {
                "id": ["a"],
                "content_quality": ["good"],
                "content_safety": ["safe"],
            }
        )
        scores = sampler.compute_scores(df)
        np.testing.assert_allclose(scores.values, [0.8 * 0.5])

    def test_normalization(self):
        config = ScoringConfig(
            columns={
                "content_quality": ColumnScoring(
                    category_scores={"excellent": 1.0, "poor": 0.2},
                    weight=1.0,
                ),
            },
            aggregation="weighted_mean",
            normalize=True,
        )
        sampler = ScoreSampler(config=config)
        df = _make_annotations_df(
            ids=["a", "b", "c"],
            quality=["excellent", "poor", "excellent"],
        )
        scores = sampler.compute_scores(df)
        assert scores.min() == 0.0
        assert scores.max() == 1.0

    def test_no_matching_columns_raises(self):
        config = ScoringConfig(
            columns={
                "nonexistent_column": ColumnScoring(
                    category_scores={"x": 1.0},
                ),
            },
        )
        sampler = ScoreSampler(config=config)
        df = _make_annotations_df(ids=["a"], quality=["good"])
        with pytest.raises(ValueError, match="No annotation columns matched"):
            sampler.compute_scores(df)

    def test_skips_missing_config_columns(self):
        """If config lists columns A and B but annotations only have A, score using A only."""
        config = ScoringConfig(
            columns={
                "content_quality": ColumnScoring(
                    category_scores={"good": 0.8},
                    weight=1.0,
                ),
                "content_safety": ColumnScoring(
                    category_scores={"safe": 1.0},
                    weight=1.0,
                ),
            },
            aggregation="weighted_mean",
            normalize=False,
        )
        sampler = ScoreSampler(config=config)
        # Only content_quality present, no content_safety column
        df = pd.DataFrame({"id": ["a"], "content_quality": ["good"]})
        scores = sampler.compute_scores(df)
        # Should score using content_quality alone
        np.testing.assert_allclose(scores.values, [0.8])


# ============================================================
# Unit tests — selection strategies
# ============================================================


class TestSelectThreshold:
    def test_keeps_scores_above_threshold(self):
        scores = np.array([0.1, 0.5, 0.7, 0.9])
        indices = ScoreSampler._select_threshold(scores, 0.5)
        np.testing.assert_array_equal(indices, [1, 2, 3])

    def test_boundary_is_inclusive(self):
        scores = np.array([0.5, 0.5, 0.4])
        indices = ScoreSampler._select_threshold(scores, 0.5)
        np.testing.assert_array_equal(indices, [0, 1])

    def test_empty_result_when_threshold_too_high(self):
        scores = np.array([0.1, 0.2, 0.3])
        indices = ScoreSampler._select_threshold(scores, 0.9)
        assert len(indices) == 0


class TestSelectProbabilistic:
    def test_reproducible_with_seed(self):
        scores = np.array([0.1, 0.5, 0.9, 0.3])
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        idx1 = ScoreSampler._select_probabilistic(scores, 10, replace=True, rng=rng1)
        idx2 = ScoreSampler._select_probabilistic(scores, 10, replace=True, rng=rng2)
        np.testing.assert_array_equal(idx1, idx2)

    def test_without_replacement_no_duplicates(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        rng = np.random.default_rng(42)
        indices = ScoreSampler._select_probabilistic(scores, 4, replace=False, rng=rng)
        assert len(set(indices)) == 4

    def test_with_replacement_allows_duplicates(self):
        scores = np.array([1.0, 0.0, 0.0, 0.0])
        rng = np.random.default_rng(42)
        indices = ScoreSampler._select_probabilistic(scores, 10, replace=True, rng=rng)
        # All samples should be index 0 since it's the only non-zero score
        np.testing.assert_array_equal(indices, np.zeros(10, dtype=int))

    def test_zero_score_never_selected(self):
        scores = np.array([0.0, 0.0, 1.0, 0.0])
        rng = np.random.default_rng(42)
        indices = ScoreSampler._select_probabilistic(scores, 5, replace=True, rng=rng)
        assert all(i == 2 for i in indices)

    def test_all_zero_raises(self):
        scores = np.array([0.0, 0.0, 0.0])
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="All scores are zero"):
            ScoreSampler._select_probabilistic(scores, 2, replace=False, rng=rng)

    def test_sampling_proportional_to_scores(self):
        """Higher-scored examples should be selected proportionally more often."""
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        expected_probs = scores / scores.sum()  # [0.1, 0.2, 0.3, 0.4]
        n_draws = 100_000
        rng = np.random.default_rng(42)
        indices = ScoreSampler._select_probabilistic(
            scores, n_draws, replace=True, rng=rng,
        )
        counts = np.bincount(indices, minlength=len(scores))
        observed_probs = counts / n_draws
        # With 100k draws, observed frequencies should be close to expected
        np.testing.assert_allclose(observed_probs, expected_probs, atol=0.01)

    def test_without_replacement_caps_n_samples(self):
        scores = np.array([0.0, 0.5, 0.0, 0.8])  # only 2 non-zero
        rng = np.random.default_rng(42)
        indices = ScoreSampler._select_probabilistic(scores, 10, replace=False, rng=rng)
        assert len(indices) == 2
        assert set(indices) == {1, 3}


# ============================================================
# Unit tests — config loading
# ============================================================


class TestScoringConfig:
    def test_from_name_default(self):
        config = ScoringConfig.from_name("default")
        assert "content_quality" in config.columns
        assert len(config.columns) == 1

    def test_from_name_propella_all(self):
        config = ScoringConfig.from_name("propella_all")
        assert len(config.columns) == 7
        assert "content_safety" in config.columns

    def test_from_name_invalid_raises(self):
        with pytest.raises(FileNotFoundError, match="No bundled config"):
            ScoringConfig.from_name("nonexistent")

    def test_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "aggregation": "min",
                    "normalize": False,
                    "missing_id_score": 0.5,
                    "columns": {
                        "content_quality": {
                            "weight": 2.0,
                            "default_score": 0.3,
                            "category_scores": {"good": 0.9, "bad": 0.1},
                        },
                    },
                },
                f,
            )
            f.flush()
            config = ScoringConfig.from_file(f.name)
        os.unlink(f.name)

        assert config.aggregation == "min"
        assert config.normalize is False
        assert config.missing_id_score == 0.5
        assert config.columns["content_quality"].weight == 2.0
        assert config.columns["content_quality"].default_score == 0.3
        assert config.columns["content_quality"].category_scores == {
            "good": 0.9,
            "bad": 0.1,
        }

    def test_default_scoring_config_loads(self):
        """DEFAULT_SCORING_CONFIG should be loaded from default.yaml at import time."""
        assert isinstance(DEFAULT_SCORING_CONFIG, ScoringConfig)
        assert "content_quality" in DEFAULT_SCORING_CONFIG.columns


# ============================================================
# Unit tests — ID match validation
# ============================================================


class TestIDMatchValidation:
    def _run_apply(self, match_rate: float, force: bool = False) -> Dataset:
        n_dataset = 100
        n_matched = int(n_dataset * match_rate)
        ds = _make_dataset(n=n_dataset)
        dataset_ids = ds["id"]

        matched_ids = dataset_ids[:n_matched]
        ann_df = _make_annotations_df(
            ids=matched_ids,
            quality=["excellent"] * n_matched,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ann_path = _write_annotations_parquet(ann_df, tmpdir)
            config = _simple_config()
            sampler = ScoreSampler(config=config)
            result, _ = sampler.apply(
                ds,
                ann_path,
                mode="threshold",
                threshold=0.0,
                force=force,
            )
            return result

    def test_below_20_raises(self):
        with pytest.raises(ValueError, match="Only 10.0% of dataset IDs"):
            self._run_apply(match_rate=0.10)

    def test_below_20_force_continues(self):
        result = self._run_apply(match_rate=0.10, force=True)
        assert len(result) > 0

    def test_between_20_and_90_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            self._run_apply(match_rate=0.50)
        assert "50.0% of dataset IDs have annotations" in caplog.text

    def test_above_90_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            self._run_apply(match_rate=0.95)
        assert "have annotations" not in caplog.text


# ============================================================
# Integration tests — full apply pipeline
# ============================================================


class TestApplyIntegration:
    def _setup(self, tmpdir: str, n: int = 50):
        ds = _make_dataset(n=n)
        ids = ds["id"]
        rng = np.random.default_rng(123)
        ann_df = _make_annotations_df(
            ids=ids,
            quality=rng.choice(ORDINAL_LABELS["content_quality"], n).tolist(),
            safety=rng.choice(ORDINAL_LABELS["content_safety"], n).tolist(),
            integrity=rng.choice(ORDINAL_LABELS["content_integrity"], n).tolist(),
            density=rng.choice(ORDINAL_LABELS["information_density"], n).tolist(),
            educational=rng.choice(ORDINAL_LABELS["educational_value"], n).tolist(),
            reasoning=rng.choice(ORDINAL_LABELS["reasoning_indicators"], n).tolist(),
            commercial=rng.choice(ORDINAL_LABELS["commercial_bias"], n).tolist(),
        )
        ann_path = _write_annotations_parquet(ann_df, tmpdir)
        return ds, ann_path, ann_df

    def test_threshold_filters_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds, ann_path, _ = self._setup(tmpdir)
            config = _simple_config()
            sampler = ScoreSampler(config=config)
            result, _ = sampler.apply(ds, ann_path, mode="threshold", threshold=0.5)
            assert len(result) < len(ds)
            assert len(result) > 0
            # All remaining IDs should exist in original
            assert all(rid in ds["id"] for rid in result["id"])

    def test_sample_without_replacement_correct_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds, ann_path, _ = self._setup(tmpdir)
            config = _simple_config()
            sampler = ScoreSampler(config=config)
            n_samples = 20
            result, _ = sampler.apply(
                ds,
                ann_path,
                mode="sample_without_replacement",
                n_samples=n_samples,
                seed=42,
            )
            assert len(result) == n_samples
            # No duplicate IDs
            assert len(set(result["id"])) == n_samples

    def test_sample_with_replacement_allows_duplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds, ann_path, _ = self._setup(tmpdir, n=5)
            config = _simple_config()
            sampler = ScoreSampler(config=config)
            result, _ = sampler.apply(
                ds,
                ann_path,
                mode="sample_with_replacement",
                n_samples=50,
                seed=42,
            )
            assert len(result) == 50
            # With 50 samples from 5 examples, must have duplicates
            assert len(set(result["id"])) < 50

    def test_propella_all_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds, ann_path, _ = self._setup(tmpdir)
            config = ScoringConfig.from_name("propella_all")
            sampler = ScoreSampler(config=config)
            result, _ = sampler.apply(ds, ann_path, mode="threshold", threshold=0.5)
            assert len(result) <= len(ds)

    def test_custom_yaml_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds, ann_path, _ = self._setup(tmpdir)

            config_path = os.path.join(tmpdir, "custom.yaml")
            with open(config_path, "w") as f:
                yaml.dump(
                    {
                        "aggregation": "weighted_mean",
                        "normalize": True,
                        "missing_id_score": 0.0,
                        "columns": {
                            "content_quality": {
                                "weight": 1.0,
                                "default_score": 0.5,
                                "category_scores": {
                                    "excellent": 1.0,
                                    "good": 0.75,
                                    "adequate": 0.4,
                                    "poor": 0.1,
                                    "unacceptable": 0.0,
                                },
                            },
                            "content_safety": {
                                "weight": 10.0,
                                "default_score": 0.0,
                                "category_scores": {
                                    "safe": 1.0,
                                    "mild_concerns": 0.5,
                                    "nsfw": 0.0,
                                    "harmful": 0.0,
                                    "illegal": 0.0,
                                },
                            },
                        },
                    },
                    f,
                )

            config = ScoringConfig.from_file(config_path)
            sampler = ScoreSampler(config=config)
            result, _ = sampler.apply(ds, ann_path, mode="threshold", threshold=0.5)
            assert len(result) <= len(ds)

    def test_output_preserves_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds, ann_path, _ = self._setup(tmpdir)
            config = _simple_config()
            sampler = ScoreSampler(config=config)
            result, _ = sampler.apply(ds, ann_path, mode="threshold", threshold=0.0)
            assert result.column_names == ds.column_names

    def test_seed_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds, ann_path, _ = self._setup(tmpdir)
            config = _simple_config()
            sampler = ScoreSampler(config=config)
            r1, _ = sampler.apply(
                ds,
                ann_path,
                mode="sample_without_replacement",
                n_samples=10,
                seed=99,
            )
            r2, _ = sampler.apply(
                ds,
                ann_path,
                mode="sample_without_replacement",
                n_samples=10,
                seed=99,
            )
            assert r1["id"] == r2["id"]
