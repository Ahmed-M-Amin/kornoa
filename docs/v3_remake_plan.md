# V3-Remake / V4-Remake Rule Search Plan

## Objective

V3-remake uses original V2B validation predictions as the base and searches controlled detector rules that can only support `0 -> 1` flips for uncertain V2B predictions. The competition metric is binary F1 for `target = 1` with `zero_division = 0`.

## Boundaries

This workflow does not train any model, does not retrain the detector, does not use V2B-Enhanced artifacts, does not use test labels, and does not submit automatically.

## Rule Search

The search combines detector confidence thresholds `[0.10, 0.15, 0.20, 0.25, 0.30, 0.35]` with V2B uncertainty margins `[0.02, 0.04, 0.06, 0.08, 0.10, 0.12]`. A detector rule is only allowed when `abs(v2b_probability - v2b_threshold) <= margin` and the original V2B prediction is `0`.

Always-reject categories support rejection, never-reject categories do not flip, and conditional categories require an area threshold. High-confidence V2B predictions remain unchanged.

## Outputs

The runner writes reports and a conservative V2B-aligned submission under `outputs/hybrid/v4_remake/`. If only the target-only original V2B test submission is available, the submission keeps original V2B test targets because test uncertainty cannot be inferred safely from hard labels.

## Commands

Compile:

```bash
python -m py_compile src/inference/v3_remake_rule_search.py
```

Run tests:

```bash
python -m pytest tests/test_v3_remake_rule_search.py -v
```

Run rule search:

```bash
python -m src.inference.v3_remake_rule_search --config configs/v3_remake_rule_search.yaml
```

Refresh file path index:

```bash
python scripts/update_file_path_index.py
```
