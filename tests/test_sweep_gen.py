"""Tests for the sweep config generator."""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from propella_curation.sweep_gen import base_category_scores, generate
from propella_curation.score_sampler import ScoringConfig


def test_geometric_constant_per_step_odds():
    tiers = ["unacceptable", "poor", "adequate", "good", "excellent"]
    s = base_category_scores(tiers, "geometric", floor=[])
    vals = [s[t] for t in tiers]
    ratios = [vals[i + 1] / vals[i] for i in range(len(vals) - 1)]
    # geometric ladder -> constant adjacent ratio (== e)
    assert all(math.isclose(r, math.e, rel_tol=1e-3) for r in ratios)
    assert math.isclose(s["excellent"], 1.0)


def test_linear_equal_interval():
    tiers = ["a", "b", "c", "d", "e"]
    s = base_category_scores(tiers, "linear", floor=[])
    assert [s[t] for t in tiers] == [0.2, 0.4, 0.6, 0.8, 1.0]


def test_floor_zeros_tiers():
    tiers = ["unacceptable", "poor", "adequate", "good", "excellent"]
    s = base_category_scores(tiers, "geometric", floor=["unacceptable", "poor"])
    assert s["unacceptable"] == 0.0 and s["poor"] == 0.0
    assert s["adequate"] > 0 and s["excellent"] == 1.0


def test_generate_emits_configs_and_manifest(tmp_path):
    spec = {
        "name": "t", "column": "content_quality",
        "tiers": ["poor", "good", "excellent"], "base_shape": "geometric",
        "floor": [], "temperatures": [0, 1, 2],
        "mode": "sample_with_replacement", "sampling": "systematic",
        "dataset_path": "/d", "annotations_path": "/a", "output_root": "/out",
    }
    spec_path = tmp_path / "t.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    res = generate(spec_path)
    assert res["n_runs"] == 3
    # run id encodes the sampling method; configs land under it
    assert res["config_dir"].endswith("/t_sys")
    cfg = ScoringConfig.from_file(Path(res["config_dir"]) / "t_sys_b2.yaml")
    assert cfg.temperature == 2.0 and cfg.normalize is False
    assert cfg.sampling == "systematic"
    manifest = json.loads(Path(res["manifest"]).read_text())
    assert [r["temperature"] for r in manifest["runs"]] == [0.0, 1.0, 2.0]
    assert manifest["runs"][0]["output_dir"] == "/out/t_sys/b0"


def test_run_id_encodes_sampling_and_floor(tmp_path):
    from propella_curation.sweep_gen import SweepSpec, run_id

    def mk(**kw):
        base = dict(name="s", column="content_quality", tiers=["a", "b"],
                    base_shape="linear", floor=[], temperatures=[1],
                    mode="sample_with_replacement", dataset_path="/d",
                    annotations_path="/a", output_root="/o")
        base.update(kw)
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump(base))
        return SweepSpec.from_file(p)

    assert run_id(mk(sampling="iid")) == "s_iid"
    assert run_id(mk(sampling="systematic")) == "s_sys"
    assert run_id(mk(sampling="systematic", floor=["a"])) == "s_sys_floored"
