"""Tests for the project file path index updater."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest


def test_file_path_index_updater_excludes_private_and_generated_paths(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "artifacts").mkdir()
    (root / "artifacts" / "private.csv").write_text("secret\n", encoding="utf-8")
    (root / "outputs").mkdir()
    (root / "outputs" / "submission.csv").write_text("image_id,target\n", encoding="utf-8")

    output = root / "docs" / "file-path-index.md"
    monkeypatch.setattr(sys, "argv", ["update_file_path_index.py", "--root", str(root), "--output", str(output)])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path("scripts/update_file_path_index.py")), run_name="__main__")
    assert exc_info.value.code == 0

    index = output.read_text(encoding="utf-8")
    assert "`src/app.py`" in index
    assert "artifacts/private.csv" not in index
    assert "outputs/submission.csv" not in index
