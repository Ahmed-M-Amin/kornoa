"""Generate a project file path index for code navigation."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT = Path("docs/file-path-index.md")
EXCLUDED_DIRS = {
    ".agents",
    ".cursor",
    ".gemini",
    ".git",
    ".opencode",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "1st-krones-vision-ai-challenge",
    "artifacts",
    "datei",
    "outputs",
}
EXCLUDED_SUFFIXES = {
    ".ckpt",
    ".engine",
    ".joblib",
    ".onnx",
    ".pth",
    ".pt",
    ".pkl",
    ".pyc",
    ".safetensors",
    ".weights",
    ".zip",
}
EXCLUDED_FILES = {"open-api.py", "opencode.json"}


def collect_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRS]
        current = Path(current_root)
        for filename in filenames:
            path = current / filename
            if filename in EXCLUDED_FILES or path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            paths.append(path.relative_to(root))
    return sorted(paths, key=lambda item: item.as_posix().lower())


def group_for(path: Path) -> str:
    first = path.parts[0] if path.parts else ""
    if first == "configs":
        return "Configuration"
    if first == "docs":
        return "Documentation"
    if first == "notebooks" or path.suffix.lower() == ".ipynb":
        return "Notebooks"
    if first == "scripts":
        return "Maintenance Scripts"
    if first == "specs":
        return "Spec Kit Artifacts"
    if first == "src":
        return "Source Code"
    if first == "tests":
        return "Tests"
    if first == ".specify":
        return "Spec Kit Tooling"
    return "Repository Root"


def render_index(paths: list[Path]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Project File Path Index",
        "",
        f"Generated: {now}",
        "",
        "Refresh command:",
        "",
        "```powershell",
        "python scripts/update_file_path_index.py",
        "```",
        "",
        "Excluded by design: private datasets, `artifacts/`, `outputs/`, model weights, caches, virtual environments, secrets, and local agent/tool state.",
        "",
    ]
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(group_for(path), []).append(path)
    for group in sorted(grouped):
        lines.extend([f"## {group}", ""])
        lines.extend(f"- `{path.as_posix()}`" for path in grouped[group])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate docs/file-path-index.md")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_index(collect_paths(root)), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
