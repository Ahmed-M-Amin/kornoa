# Quickstart: Project Foundation

## Goal

Verify that SPEC-001 creates a safe repository foundation without tracking
private dataset files or generated private artifacts.

## Steps

1. Confirm the active feature:

   ```powershell
   git branch --show-current
   ```

   Expected: `001-project-foundation`

2. Inspect the planned structure:

   ```powershell
   Get-ChildItem -Force
   Get-ChildItem -Force configs
   Get-ChildItem -Force src
   Get-ChildItem -Force outputs
   Get-ChildItem -Force tests
   ```

   Expected: foundation directories and safe placeholder files exist.

3. Install dependencies after `requirements.txt` exists:

   ```powershell
   python -m pip install -r requirements.txt
   ```

   Expected: dependencies install in the active environment.

4. Run foundation tests:

   ```powershell
   pytest tests/test_project_foundation.py -q
   ```

   Expected: all foundation checks pass.

5. Verify private/generated files are ignored:

   ```powershell
   git ls-files
   git status --short --ignored
   ```

   Expected: `git ls-files` does not list private dataset files from
   `1st-krones-vision-ai-challenge/` or generated artifact contents. Private
   dataset files and generated artifact contents are ignored; safe placeholders
   remain visible when newly created.

## Success

The foundation is ready when contributors can locate configs, source packages,
tests, notebooks, generated-output categories, and documentation, while private
dataset files and generated artifacts remain protected by `.gitignore`.
