#!/usr/bin/env bash
# Curation runbook for Dolci-Instruct-SFT-Decont.
#
# This is the tracked record AND reproducer of every curated dataset in the
# family. It launches SLURM jobs via submit_curation.sh; each output dir gets a
# dataset card (README.md) with full per-dataset provenance. See docs/math.md
# for the rationale behind each choice.
#
# Baseline for comparison = the uncurated full Dolci-Instruct-SFT (each row once).
#
# NOTE: boost_qos_dbg allows only 2 concurrent jobs. The temperature sweep has 5
# points, so either launch its submit_all in batches, raise the QOS in the spec's
# slurm block (e.g. qos: boost_qos_bprod), or run the per-point commands a few at
# a time. The single cuts/downsample below are fine on dbg.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repo root

DS=/leonardo_work/OELLM_prod2026/users/knikolao/propella_annotation/data/Dolci-Instruct-SFT-Decont
ANN=/leonardo_work/OELLM_prod2026/users/knikolao/propella_annotation/output/Dolci-Instruct-SFT-annotations/shard000000.parquet
OUT=/leonardo_work/OELLM_prod2026/users/knikolao/propella_curation/Dolci-Instruct-SFT-Decont
SLURM="--writer load --time 00:30:00 --mem 120G --cpus 8"

# ── 1. Temperature sweep (geometric base, systematic resampling) ───────────────
#    beta in {0, 0.5, 1, 2, 4}, n = N (size preserved). Configs carry temperature
#    + sampling. Generated artifacts are gitignored (reproducible from the spec).
#    Output: $OUT/temp_geometric_sys/b{0,0.5,1,2,4}/
.venv/bin/propella-gen-sweep sweeps/temp_geometric.yaml
bash sweeps/temp_geometric_sys.submit_all.sh   # 5 jobs — mind the dbg 2-job cap

# ── 2. Hard cuts (threshold; each surviving row once, no resampling) ───────────
#    2a. keep >= good  (excellent + good)  -> ~96.3% kept
bash submit_curation.sh \
  --dataset-path "$DS" --annotations-path "$ANN" \
  --output-dir "$OUT/cutoff_content_quality/min_good" \
  --mode threshold --threshold 0.5 \
  --config content_quality_min_good \
  $SLURM

#    2b. keep only excellent  -> ~29.9% kept
bash submit_curation.sh \
  --dataset-path "$DS" --annotations-path "$ANN" \
  --output-dir "$OUT/cutoff_content_quality/min_excellent" \
  --mode threshold --threshold 0.5 \
  --config content_quality_min_excellent \
  $SLURM

# ── 3. Weighted downselect (without replacement; no duplicates) ────────────────
#    all excellent + ~50% of good (config floors below good, temperature=4 in the
#    config prioritizes excellent). n = 643,186 + 0.5*1,428,695 = 1,357,534 (~63%).
#    Output: $OUT/downsample_geometric/excellent_halfgood/
bash submit_curation.sh \
  --dataset-path "$DS" --annotations-path "$ANN" \
  --output-dir "$OUT/downsample_geometric/excellent_halfgood" \
  --mode sample_without_replacement \
  --config content_quality_min_good_graded \
  --n-samples 1357534 \
  $SLURM
