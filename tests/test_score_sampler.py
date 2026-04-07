"""Unit and integration tests for ScoreSampler."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
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
            scores,
            n_draws,
            replace=True,
            rng=rng,
        )
        counts = np.bincount(indices, minlength=len(scores))
        observed_probs = counts / n_draws
        # With 100k draws, observed frequencies should be close to expected
        np.testing.assert_allclose(observed_probs, expected_probs, atol=0.01)

    def test_max_duplications_respected(self):
        scores = np.array([0.1, 0.5, 0.9, 0.0, 0.7, 0.3])
        rng = np.random.default_rng(0)
        indices = ScoreSampler._select_probabilistic(
            scores,
            n_samples=200,
            replace=True,
            rng=rng,
            max_duplications=3,
        )
        counts = np.bincount(indices, minlength=len(scores))
        assert counts.max() <= 3
        # zero-score example never selected
        assert counts[3] == 0

    def test_max_duplications_caps_infeasible_request(self):
        scores = np.array([0.1, 0.5, 0.9, 0.0, 0.7])  # 4 nonzero
        rng = np.random.default_rng(0)
        indices = ScoreSampler._select_probabilistic(
            scores,
            n_samples=100,
            replace=True,
            rng=rng,
            max_duplications=2,
        )
        # capacity is 4 nonzero * 2 = 8
        assert len(indices) == 8
        counts = np.bincount(indices, minlength=len(scores))
        assert counts.max() <= 2

    def test_max_duplications_none_matches_uncapped(self):
        scores = np.array([0.1, 0.5, 0.9, 0.3])
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        idx1 = ScoreSampler._select_probabilistic(
            scores,
            n_samples=20,
            replace=True,
            rng=rng1,
        )
        idx2 = ScoreSampler._select_probabilistic(
            scores,
            n_samples=20,
            replace=True,
            rng=rng2,
            max_duplications=None,
        )
        np.testing.assert_array_equal(idx1, idx2)

    def test_max_duplications_invalid_raises(self):
        scores = np.array([0.1, 0.5, 0.9])
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="max_duplications must be >= 1"):
            ScoreSampler._select_probabilistic(
                scores,
                n_samples=5,
                replace=True,
                rng=rng,
                max_duplications=0,
            )

    def test_max_duplications_one_equivalent_to_without_replacement(self):
        scores = np.array([0.1, 0.5, 0.9, 0.0, 0.7, 0.3])
        rng = np.random.default_rng(0)
        indices = ScoreSampler._select_probabilistic(
            scores,
            n_samples=5,
            replace=True,
            rng=rng,
            max_duplications=1,
        )
        # 5 nonzero examples, max_dup=1 → exactly 5 unique indices, no zero-score idx
        assert len(indices) == 5
        assert len(set(indices)) == 5
        assert 3 not in indices

    def test_max_duplications_single_nonzero_score(self):
        scores = np.array([0.0, 0.0, 0.5, 0.0])
        rng = np.random.default_rng(0)
        indices = ScoreSampler._select_probabilistic(
            scores,
            n_samples=5,
            replace=True,
            rng=rng,
            max_duplications=2,
        )
        # capacity = 1 nonzero × 2 = 2
        assert len(indices) == 2
        assert all(i == 2 for i in indices)

    def test_max_duplications_n_samples_zero(self):
        scores = np.array([0.1, 0.5, 0.9])
        rng = np.random.default_rng(0)
        indices = ScoreSampler._select_probabilistic(
            scores,
            n_samples=0,
            replace=True,
            rng=rng,
            max_duplications=2,
        )
        assert len(indices) == 0

    @pytest.mark.parametrize(
        "scores,n_samples,max_dup",
        [
            # moderate saturation
            (np.array([0.4, 0.3, 0.2, 0.1]), 6, 2),
            # near-saturated: capacity=8, n=7 → forces multi-saturation per batch
            (np.array([0.4, 0.3, 0.2, 0.1]), 7, 2),
            # skewed weights, near-saturated: capacity=6, n=5
            (np.array([0.6, 0.2, 0.15, 0.05]), 5, 2),
        ],
    )
    def test_max_duplications_marginals_match_rejection_baseline(
        self,
        scores,
        n_samples,
        max_dup,
    ):
        """Refill loop should produce the same marginal distribution as brute-force rejection sampling."""
        n_trials = 8000

        def rejection_sample(seed: int) -> np.ndarray:
            r = np.random.default_rng(seed)
            probs = scores / scores.sum()
            counts = np.zeros(len(scores), dtype=np.int64)
            drawn = 0
            while drawn < n_samples:
                i = r.choice(len(scores), p=probs)
                if counts[i] < max_dup:
                    counts[i] += 1
                    drawn += 1
            return counts

        ref_counts = np.zeros(len(scores))
        impl_counts = np.zeros(len(scores))
        for s in range(n_trials):
            ref_counts += rejection_sample(s)
            r = np.random.default_rng(s + 100000)
            idx = ScoreSampler._select_probabilistic(
                scores,
                n_samples=n_samples,
                replace=True,
                rng=r,
                max_duplications=max_dup,
            )
            impl_counts += np.bincount(idx, minlength=len(scores))

        ref_freq = ref_counts / ref_counts.sum()
        impl_freq = impl_counts / impl_counts.sum()
        np.testing.assert_allclose(impl_freq, ref_freq, atol=0.01)

    def test_max_duplications_reproducible_with_seed(self):
        scores = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        idx1 = ScoreSampler._select_probabilistic(
            scores,
            n_samples=30,
            replace=True,
            rng=rng1,
            max_duplications=4,
        )
        idx2 = ScoreSampler._select_probabilistic(
            scores,
            n_samples=30,
            replace=True,
            rng=rng2,
            max_duplications=4,
        )
        np.testing.assert_array_equal(idx1, idx2)

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
            result, _, _ = sampler.apply(
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
            result, _, _ = sampler.apply(ds, ann_path, mode="threshold", threshold=0.5)
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
            result, _, _ = sampler.apply(
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
            result, _, _ = sampler.apply(
                ds,
                ann_path,
                mode="sample_with_replacement",
                n_samples=50,
                seed=42,
            )
            assert len(result) == 50
            # With 50 samples from 5 examples, must have duplicates
            assert len(set(result["id"])) < 50

    def test_sample_with_replacement_max_duplications(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds, ann_path, _ = self._setup(tmpdir, n=10)
            config = _simple_config()
            sampler = ScoreSampler(config=config)
            result, _, _ = sampler.apply(
                ds,
                ann_path,
                mode="sample_with_replacement",
                n_samples=100,
                seed=42,
                max_duplications=3,
            )
            from collections import Counter

            counts = Counter(result["id"])
            assert max(counts.values()) <= 3

    def test_max_duplications_rejected_for_non_replacement_modes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds, ann_path, _ = self._setup(tmpdir, n=10)
            sampler = ScoreSampler(config=_simple_config())
            with pytest.raises(ValueError, match="max_duplications is only valid"):
                sampler.apply(
                    ds,
                    ann_path,
                    mode="threshold",
                    threshold=0.5,
                    max_duplications=3,
                )
            with pytest.raises(ValueError, match="max_duplications is only valid"):
                sampler.apply(
                    ds,
                    ann_path,
                    mode="sample_without_replacement",
                    n_samples=5,
                    max_duplications=3,
                )

    def test_dataset_card_writes_with_none_n_samples(self):
        """Regression: writing a card with n_samples=None must not raise."""
        from propella_curation.dataset_card import CurationInfo, write_dataset_card

        with tempfile.TemporaryDirectory() as tmpdir:
            info = CurationInfo(
                name="x",
                source_dataset="src",
                annotations_path="ann",
                config_name="default",
                mode="sample_with_replacement",
                threshold=None,
                n_samples=None,  # user didn't pass --n_samples
                seed=0,
                source_rows=100,
                selected_rows=100,
                scores_after=np.array([0.1, 0.5, 0.9]),
                max_duplications=3,
            )
            write_dataset_card(info, tmpdir)
            content = (Path(tmpdir) / "README.md").read_text()
            assert "n=100" in content  # falls back to source_rows
            assert "max_duplications=3" in content

    def test_curation_info_constructs_with_max_duplications(self):
        from propella_curation.dataset_card import CurationInfo

        info = CurationInfo(
            name="x",
            source_dataset="src",
            annotations_path="ann",
            config_name="default",
            mode="sample_with_replacement",
            threshold=None,
            n_samples=10,
            seed=0,
            source_rows=100,
            selected_rows=10,
            scores_after=np.array([0.1, 0.5, 0.9]),
            max_duplications=3,
        )
        assert info.max_duplications == 3
        info2 = CurationInfo(
            name="x",
            source_dataset="src",
            annotations_path="ann",
            config_name="default",
            mode="threshold",
            threshold=0.5,
            n_samples=None,
            seed=0,
            source_rows=100,
            selected_rows=10,
            scores_after=np.array([0.1, 0.5, 0.9]),
        )
        assert info2.max_duplications is None

    def test_propella_all_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds, ann_path, _ = self._setup(tmpdir)
            config = ScoringConfig.from_name("propella_all")
            sampler = ScoreSampler(config=config)
            result, _, _ = sampler.apply(ds, ann_path, mode="threshold", threshold=0.5)
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
            result, _, _ = sampler.apply(ds, ann_path, mode="threshold", threshold=0.5)
            assert len(result) <= len(ds)

    def test_output_preserves_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds, ann_path, _ = self._setup(tmpdir)
            config = _simple_config()
            sampler = ScoreSampler(config=config)
            result, _, _ = sampler.apply(ds, ann_path, mode="threshold", threshold=0.0)
            assert result.column_names == ds.column_names

    def test_seed_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds, ann_path, _ = self._setup(tmpdir)
            config = _simple_config()
            sampler = ScoreSampler(config=config)
            r1, _, _ = sampler.apply(
                ds,
                ann_path,
                mode="sample_without_replacement",
                n_samples=10,
                seed=99,
            )
            r2, _, _ = sampler.apply(
                ds,
                ann_path,
                mode="sample_without_replacement",
                n_samples=10,
                seed=99,
            )
            assert r1["id"] == r2["id"]


# ============================================================
# Integration tests — full CLI writer (load + stream)
# ============================================================


class TestWriterCLI:
    """Run the full CLI end-to-end against on-disk parquet for both writer modes.

    This is the only test that exercises the per-source-chunk writer with a
    real chunked parquet source. Catches any divergence between the load and
    stream paths and verifies max_duplications round-trips through to disk.
    """

    def _make_source_parquets(
        self, tmpdir: str, n_files: int = 3, rows_per_file: int = 40
    ) -> str:
        """Write a multi-shard source dataset with a nested messages column."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        data_dir = os.path.join(tmpdir, "source", "data")
        os.makedirs(data_dir)
        for f in range(n_files):
            start = f * rows_per_file
            ids = [f"d{start + i:04d}" for i in range(rows_per_file)]
            messages = [
                [
                    {"role": "user", "content": f"q{start + i}"},
                    {"role": "assistant", "content": f"a{start + i}"},
                ]
                for i in range(rows_per_file)
            ]
            table = pa.table({"id": ids, "messages": messages})
            pq.write_table(table, os.path.join(data_dir, f"train-{f:05d}.parquet"))
        return os.path.join(tmpdir, "source")

    def _run_cli(self, src_dir: str, ann_path: str, out_dir: str, writer: str) -> None:
        from propella_curation.score_sampler import main

        argv = [
            "propella-score-sampler",
            "--dataset_path",
            src_dir,
            "--annotations_path",
            ann_path,
            "--output_dir",
            out_dir,
            "--mode",
            "sample_with_replacement",
            "--n_samples",
            "200",
            "--max_duplications",
            "3",
            "--seed",
            "7",
            "--writer",
            writer,
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            main()
        finally:
            sys.argv = old_argv

    def _read_output_ids(self, out_dir: str) -> list[str]:
        import glob

        import pyarrow.parquet as pq

        files = sorted(glob.glob(os.path.join(out_dir, "data", "*.parquet")))
        ids: list[str] = []
        for f in files:
            ids.extend(pq.read_table(f, columns=["id"])["id"].to_pylist())
        return ids

    def test_load_and_stream_produce_same_multiset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = self._make_source_parquets(tmpdir, n_files=3, rows_per_file=40)
            ids = [f"d{i:04d}" for i in range(120)]
            ann_df = _make_annotations_df(
                ids=ids,
                quality=["excellent"] * 60 + ["good"] * 40 + ["poor"] * 20,
            )
            ann_path = os.path.join(tmpdir, "ann.parquet")
            ann_df.to_parquet(ann_path)

            load_dir = os.path.join(tmpdir, "out_load")
            stream_dir = os.path.join(tmpdir, "out_stream")
            self._run_cli(src_dir, ann_path, load_dir, "load")
            self._run_cli(src_dir, ann_path, stream_dir, "stream")

            load_ids = self._read_output_ids(load_dir)
            stream_ids = self._read_output_ids(stream_dir)

            # Same row count
            assert len(load_ids) == len(stream_ids) == 200

            # Same multiset of rows (different ordering is allowed)
            from collections import Counter

            assert Counter(load_ids) == Counter(stream_ids)

            # max_duplications respected
            assert max(Counter(load_ids).values()) <= 3
            assert max(Counter(stream_ids).values()) <= 3

            # 'load' preserves sampler order; 'stream' sorts by source row.
            # IDs are zero-padded so lex order == source order, making the
            # sortedness check meaningful here.
            assert stream_ids == sorted(stream_ids)
            assert load_ids != sorted(load_ids)  # extremely unlikely to be sorted

            # Output schema must equal source schema (no silent type drift).
            import glob
            import pyarrow.parquet as pq
            src_schema = pq.read_schema(
                sorted(glob.glob(os.path.join(src_dir, "data", "*.parquet")))[0]
            )
            for out_dir in (load_dir, stream_dir):
                for f in sorted(glob.glob(os.path.join(out_dir, "data", "*.parquet"))):
                    assert pq.read_schema(f).equals(src_schema, check_metadata=False)

    def test_apply_returns_source_table_indices_after_select(self):
        """If the input dataset already has an indices mapping, apply()'s
        returned indices must reference dataset.data.table directly (composed),
        so a caller can take from data.table without further composition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds, ann_path, _ = self._setup_ds_with_annotations(tmpdir)
            shuffled = ds.shuffle(seed=11)  # creates an _indices mapping
            assert shuffled._indices is not None

            sampler = ScoreSampler(config=_simple_config())
            filtered, _, gather = sampler.apply(
                shuffled,
                ann_path,
                mode="sample_with_replacement",
                n_samples=20,
                seed=3,
                max_duplications=2,
            )
            # gather should index into shuffled.data.table directly
            taken = shuffled.data.table.take(pa.array(gather)).column("id").to_pylist()
            assert taken == filtered["id"]

    def test_load_and_stream_match_with_shuffled_input(self):
        """Same as the main multiset test but the source dataset is shuffled
        first via the CLI's normal load path. This exercises the dataset_card
        write step too."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = self._make_source_parquets(tmpdir, n_files=4, rows_per_file=30)
            ids = [f"d{i:04d}" for i in range(120)]
            ann_df = _make_annotations_df(
                ids=ids,
                quality=["excellent"] * 80 + ["good"] * 30 + ["poor"] * 10,
            )
            ann_path = os.path.join(tmpdir, "ann.parquet")
            ann_df.to_parquet(ann_path)

            load_dir = os.path.join(tmpdir, "out_load")
            stream_dir = os.path.join(tmpdir, "out_stream")
            self._run_cli(src_dir, ann_path, load_dir, "load")
            self._run_cli(src_dir, ann_path, stream_dir, "stream")

            load_ids = self._read_output_ids(load_dir)
            stream_ids = self._read_output_ids(stream_dir)
            from collections import Counter
            assert Counter(load_ids) == Counter(stream_ids)

    def _setup_ds_with_annotations(self, tmpdir: str):
        src_dir = self._make_source_parquets(tmpdir, n_files=3, rows_per_file=20)
        from datasets import load_dataset
        import glob as _g
        files = sorted(_g.glob(os.path.join(src_dir, "data", "*.parquet")))
        ds = load_dataset("parquet", data_files=files, split="train")
        ann_df = _make_annotations_df(
            ids=ds["id"],
            quality=["excellent"] * len(ds),
        )
        ann_path = os.path.join(tmpdir, "ann.parquet")
        ann_df.to_parquet(ann_path)
        return ds, ann_path, ann_df
