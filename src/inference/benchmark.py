"""V1 inference benchmark reporting for SPEC-006."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

from src.inference.predict import DEFAULT_ARTIFACT_ROOT, DEFAULT_V2_ARTIFACT_ROOT, discover_images, predict_images, resolve_v1_artifact_paths, resolve_v2_artifact_paths


DEFAULT_BENCHMARK_OUTPUT = Path("outputs/benchmarks/v1_inference_benchmark.json")
DEFAULT_V2_BENCHMARK_OUTPUT = Path("outputs/kaggle_v2/benchmarks/v2_inference_benchmark.json")


@dataclass(frozen=True)
class BenchmarkReport:
    total_images: int
    total_time_seconds: float
    average_time_per_image: float
    images_per_second: float
    batch_size: int
    device: str
    model_name: str
    image_size: int
    artifact_root: str
    speed_multiplier_vs_v1: float | None = None
    within_v2_speed_ceiling: bool | None = None


def run_benchmark(
    *,
    image_dir: str | Path,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    model_path: str | Path | None = None,
    threshold_path: str | Path | None = None,
    output_path: str | Path = DEFAULT_BENCHMARK_OUTPUT,
    model_name: str = "efficientnet_b0",
    synthetic_smoke: bool = False,
    device: str = "auto",
    batch_size: int = 1,
    image_size: int = 384,
    v1_average_time_per_image: float | None = None,
    speed_ceiling_multiplier: float = 2.0,
) -> BenchmarkReport:
    """Run V1 prediction and save benchmark metrics."""

    artifacts = resolve_v2_artifact_paths(artifact_root) if _looks_like_v2_benchmark(artifact_root, output_path) else resolve_v1_artifact_paths(artifact_root)
    image_paths = discover_images(image_dir)
    started = time.perf_counter()
    predict_images(
        image_paths,
        artifact_root=artifacts.artifact_root,
        model_path=model_path,
        threshold_path=threshold_path,
        model_name=model_name,
        synthetic_smoke=synthetic_smoke,
        device=device,
        batch_size=batch_size,
        image_size=image_size,
    )
    total = max(0.0, time.perf_counter() - started)
    count = len(image_paths)
    average = total / count if count else 0.0
    ips = count / total if total > 0 else 0.0
    speed_multiplier = (average / v1_average_time_per_image) if v1_average_time_per_image and v1_average_time_per_image > 0 else None
    report = BenchmarkReport(
        total_images=count,
        total_time_seconds=round(total, 6),
        average_time_per_image=round(average, 6),
        images_per_second=round(ips, 6),
        batch_size=batch_size,
        device=device,
        model_name=model_name,
        image_size=image_size,
        artifact_root=str(artifacts.artifact_root),
        speed_multiplier_vs_v1=None if speed_multiplier is None else round(speed_multiplier, 6),
        within_v2_speed_ceiling=None if speed_multiplier is None else speed_multiplier <= speed_ceiling_multiplier,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report


def _looks_like_v2_benchmark(artifact_root: str | Path, output_path: str | Path) -> bool:
    root = Path(artifact_root)
    out = Path(output_path)
    return "kaggle_v2" in root.parts or "kaggle_v2" in out.parts or root == DEFAULT_V2_ARTIFACT_ROOT


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark SPEC-006 V1 inference.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--threshold-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_BENCHMARK_OUTPUT)
    parser.add_argument("--model-name", default="efficientnet_b0")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=384)
    args = parser.parse_args(argv)

    report = run_benchmark(
        image_dir=args.image_dir,
        artifact_root=args.artifact_root,
        model_path=args.model_path,
        threshold_path=args.threshold_path,
        output_path=args.output_path,
        model_name=args.model_name,
        synthetic_smoke=args.synthetic_smoke,
        device=args.device,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    print(f"Benchmarked {report.total_images} images at {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
