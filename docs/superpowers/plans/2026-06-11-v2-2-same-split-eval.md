# V2.2 Same-Split Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a validation-only evaluator that compares V2.2 against V2B on the original locked V2B validation rows and writes the required reports and error splits.

**Architecture:** Add a dedicated analysis module that loads the V2.2 config, resolves the locked V2B validation rows, runs V2.2 inference on those same image IDs, computes sectioned metrics, and writes analysis artifacts under an isolated output root. Reuse the existing inference and hard-example utilities instead of touching training or submission flows.

**Tech Stack:** Python 3.11, pandas, PyYAML, pytest, scikit-learn, existing project inference/training helpers

---

### Task 1: Add failing evaluator tests

**Files:**
- Create: `tests/test_v2_2_same_split_eval.py`

- [ ] **Step 1: Write failing tests for same-row metrics, hard-example sections, safety checks, and CLI behavior**
- [ ] **Step 2: Run `python -m pytest tests/test_v2_2_same_split_eval.py -q` and verify failure because `src.analysis.v2_2_same_split_eval` does not exist**

### Task 2: Implement the evaluator module

**Files:**
- Create: `src/analysis/v2_2_same_split_eval.py`

- [ ] **Step 1: Add config loading, safety validation, V2B row loading, hard-example tagging, metric helpers, report writing, and CLI entrypoint**
- [ ] **Step 2: Re-run `python -m pytest tests/test_v2_2_same_split_eval.py -q` and fix failures until green**

### Task 3: Verify end-to-end requirements

**Files:**
- Verify: `src/analysis/v2_2_same_split_eval.py`
- Verify: `tests/test_v2_2_same_split_eval.py`

- [ ] **Step 1: Run `python -m py_compile src/analysis/v2_2_same_split_eval.py`**
- [ ] **Step 2: Run `python -m pytest tests -q`**
- [ ] **Step 3: Summarize changed files, test results, Kaggle command, and confirm no training and no submission**
