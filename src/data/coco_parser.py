"""COCO annotation parser for the Krones Vision AI project.

Parses COCO-style annotation files, validates required collections,
normalizes records into typed structures, and provides diagnostic summaries.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Error / diagnostic structures (T005)
# ---------------------------------------------------------------------------


class CocoParseError(Exception):
    """Raised when COCO annotation parsing or validation fails."""

    def __init__(self, message: str, *, file_path: Optional[Union[str, Path]] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.file_path = file_path
        self.details = details or {}


@dataclass
class CocoDiagnostic:
    """A single diagnostic message emitted during parsing or validation."""

    severity: str  # "error" | "warning"
    code: str      # e.g. "missing_collection", "orphaned_annotation", "malformed_bbox"
    message: str
    context: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Normalized record structures (T006)
# ---------------------------------------------------------------------------


@dataclass
class ImageRecord:
    """Normalized image record."""

    image_id: Union[int, str]
    file_name: str
    width: int
    height: int
    source: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class CategoryRecord:
    """Normalized category / defect label record."""

    category_id: Union[int, str]
    name: str
    supercategory: Optional[str] = None
    source: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class AnnotationRecord:
    """Normalized annotation record."""

    annotation_id: Union[int, str]
    image_id: Union[int, str]
    category_id: Union[int, str]
    bbox: Optional[Tuple[float, float, float, float]] = None
    area: Optional[float] = None
    segmentation: Optional[Any] = None
    iscrowd: int = 0
    source: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class CocoSummary:
    """Summary of parsed COCO data and validation results."""

    image_count: int = 0
    annotation_count: int = 0
    category_count: int = 0
    images_with_annotations: int = 0
    images_without_annotations: int = 0
    orphaned_annotations: int = 0
    malformed_bboxes: int = 0
    unknown_categories: int = 0
    duplicate_ids: Dict[str, int] = field(default_factory=dict)
    diagnostics: List[CocoDiagnostic] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)


@dataclass
class CocoParseResult:
    """Result of parsing a COCO annotation file."""

    images: List[ImageRecord] = field(default_factory=list)
    annotations: List[AnnotationRecord] = field(default_factory=list)
    categories: List[CategoryRecord] = field(default_factory=list)
    image_to_annotations: Dict[Union[int, str], List[AnnotationRecord]] = field(default_factory=dict)
    category_to_annotations: Dict[Union[int, str], List[AnnotationRecord]] = field(default_factory=dict)
    summary: CocoSummary = field(default_factory=CocoSummary)

    def category_distribution(self) -> Dict[Union[int, str], int]:
        """Return count of annotations per category."""
        dist: Dict[Union[int, str], int] = {}
        for ann in self.annotations:
            dist[ann.category_id] = dist.get(ann.category_id, 0) + 1
        return dist

    def annotated_images(self) -> List[ImageRecord]:
        """Return images that have at least one annotation."""
        annotated_ids = set(self.image_to_annotations.keys())
        return [img for img in self.images if img.image_id in annotated_ids]

    def unannotated_images(self) -> List[ImageRecord]:
        """Return images with no annotations."""
        annotated_ids = set(self.image_to_annotations.keys())
        return [img for img in self.images if img.image_id not in annotated_ids]

    def roi_available_count(self) -> int:
        """Count annotations with either bbox or segmentation data."""
        return sum(
            1
            for ann in self.annotations
            if ann.bbox is not None or ann.segmentation is not None
        )

    def segmentation_available_count(self) -> int:
        """Count annotations with segmentation data."""
        return sum(
            1
            for ann in self.annotations
            if ann.segmentation is not None and ann.segmentation
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_id(value: Any) -> Union[int, str]:
    """Keep numeric IDs as int, otherwise string."""
    if isinstance(value, int):
        return value
    return str(value)


def _read_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Read and parse a JSON file with readable diagnostics."""
    path = Path(path)
    if not path.exists():
        raise CocoParseError(
            f"Annotation file not found: {path.name}",
            file_path=path,
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as exc:
        raise CocoParseError(
            f"Cannot read annotation file: {path.name}",
            file_path=path,
            details={"reason": str(exc)},
        ) from exc

    if not raw.strip():
        raise CocoParseError(
            f"Annotation file is empty: {path.name}",
            file_path=path,
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CocoParseError(
            f"Malformed JSON in annotation file: {path.name}",
            file_path=path,
            details={"reason": str(exc), "line": exc.lineno, "column": exc.colno},
        ) from exc

    if not isinstance(data, dict):
        raise CocoParseError(
            f"Annotation file must contain a JSON object, got {type(data).__name__}",
            file_path=path,
        )

    return data


def _validate_required_collections(data: Dict[str, Any], file_path: Optional[Path] = None) -> List[CocoDiagnostic]:
    """Ensure required top-level collections exist (T016)."""
    diagnostics: List[CocoDiagnostic] = []
    required = ("images", "annotations", "categories")
    for key in required:
        if key not in data:
            diagnostics.append(
                CocoDiagnostic(
                    severity="error",
                    code="missing_collection",
                    message=f"Required collection '{key}' is missing from the annotation file",
                    context={"collection": key},
                )
            )
        elif not isinstance(data[key], list):
            diagnostics.append(
                CocoDiagnostic(
                    severity="error",
                    code="invalid_collection_type",
                    message=f"Collection '{key}' must be a list, got {type(data[key]).__name__}",
                    context={"collection": key},
                )
            )
    return diagnostics


def _normalize_image(raw: Dict[str, Any]) -> ImageRecord:
    return ImageRecord(
        image_id=_coerce_id(raw.get("id", raw.get("image_id"))),
        file_name=str(raw.get("file_name", "")),
        width=int(raw.get("width", 0)),
        height=int(raw.get("height", 0)),
        source=raw,
    )


def _normalize_category(raw: Dict[str, Any]) -> CategoryRecord:
    return CategoryRecord(
        category_id=_coerce_id(raw.get("id", raw.get("category_id"))),
        name=str(raw.get("name", "")),
        supercategory=raw.get("supercategory"),
        source=raw,
    )


def _normalize_annotation(raw: Dict[str, Any]) -> AnnotationRecord:
    bbox = raw.get("bbox")
    if bbox is not None:
        try:
            bbox = tuple(float(v) for v in bbox)
        except (TypeError, ValueError):
            bbox = None
    return AnnotationRecord(
        annotation_id=_coerce_id(raw.get("id", raw.get("annotation_id"))),
        image_id=_coerce_id(raw.get("image_id")),
        category_id=_coerce_id(raw.get("category_id")),
        bbox=bbox if bbox is None or len(bbox) == 4 else None,
        area=raw.get("area"),
        segmentation=raw.get("segmentation"),
        iscrowd=int(raw.get("iscrowd", 0)),
        source=raw,
    )


# ---------------------------------------------------------------------------
# Relationship validation helpers (used by US2 tests, implemented here for completeness)
# ---------------------------------------------------------------------------


def _validate_relationships(
    images: List[ImageRecord],
    annotations: List[AnnotationRecord],
    categories: List[CategoryRecord],
) -> List[CocoDiagnostic]:
    diagnostics: List[CocoDiagnostic] = []
    image_ids: Set[Union[int, str]] = {img.image_id for img in images}
    category_ids: Set[Union[int, str]] = {cat.category_id for cat in categories}

    # Orphaned annotations
    for ann in annotations:
        if ann.image_id not in image_ids:
            diagnostics.append(
                CocoDiagnostic(
                    severity="warning",
                    code="orphaned_annotation",
                    message=f"Annotation {ann.annotation_id} references unknown image {ann.image_id}",
                    context={"annotation_id": ann.annotation_id, "image_id": ann.image_id},
                )
            )
        if ann.category_id not in category_ids:
            diagnostics.append(
                CocoDiagnostic(
                    severity="warning",
                    code="unknown_category",
                    message=f"Annotation {ann.annotation_id} references unknown category {ann.category_id}",
                    context={"annotation_id": ann.annotation_id, "category_id": ann.category_id},
                )
            )
        raw_bbox = ann.source.get("bbox")
        has_bbox_key = "bbox" in ann.source
        malformed_bbox = False
        if has_bbox_key and (
            raw_bbox is None
            or not isinstance(raw_bbox, list)
            or len(raw_bbox) != 4
            or any(not isinstance(v, (int, float)) for v in raw_bbox)
        ):
            malformed_bbox = True
        elif ann.bbox is not None:
            x, y, w, h = ann.bbox
            malformed_bbox = x < 0 or y < 0 or w <= 0 or h <= 0

        if malformed_bbox:
            diagnostics.append(
                CocoDiagnostic(
                    severity="warning",
                    code="malformed_bbox",
                    message=f"Annotation {ann.annotation_id} has invalid bbox {raw_bbox}",
                    context={"annotation_id": ann.annotation_id, "bbox": raw_bbox},
                )
            )

    return diagnostics


def _detect_duplicate_ids(
    images: List[ImageRecord],
    annotations: List[AnnotationRecord],
    categories: List[CategoryRecord],
) -> Tuple[List[CocoDiagnostic], Dict[str, int]]:
    """Detect duplicate identifiers and return diagnostics + duplicate count map."""
    diagnostics: List[CocoDiagnostic] = []
    duplicate_counts: Dict[str, int] = {}

    # Check image IDs
    image_id_counts: Dict[Union[int, str], int] = {}
    for img in images:
        image_id_counts[img.image_id] = image_id_counts.get(img.image_id, 0) + 1
    duplicates = {k: v for k, v in image_id_counts.items() if v > 1}
    for dup_id, count in duplicates.items():
        diagnostics.append(
            CocoDiagnostic(
                severity="warning",
                code="duplicate_image_id",
                message=f"Duplicate image ID '{dup_id}' appears {count} times",
                context={"image_id": dup_id, "count": count},
            )
        )
    if duplicates:
        duplicate_counts["images"] = len(duplicates)

    # Check annotation IDs
    ann_id_counts: Dict[Union[int, str], int] = {}
    for ann in annotations:
        ann_id_counts[ann.annotation_id] = ann_id_counts.get(ann.annotation_id, 0) + 1
    duplicates = {k: v for k, v in ann_id_counts.items() if v > 1}
    for dup_id, count in duplicates.items():
        diagnostics.append(
            CocoDiagnostic(
                severity="warning",
                code="duplicate_annotation_id",
                message=f"Duplicate annotation ID '{dup_id}' appears {count} times",
                context={"annotation_id": dup_id, "count": count},
            )
        )
    if duplicates:
        duplicate_counts["annotations"] = len(duplicates)

    # Check category IDs
    cat_id_counts: Dict[Union[int, str], int] = {}
    for cat in categories:
        cat_id_counts[cat.category_id] = cat_id_counts.get(cat.category_id, 0) + 1
    duplicates = {k: v for k, v in cat_id_counts.items() if v > 1}
    for dup_id, count in duplicates.items():
        diagnostics.append(
            CocoDiagnostic(
                severity="warning",
                code="duplicate_category_id",
                message=f"Duplicate category ID '{dup_id}' appears {count} times",
                context={"category_id": dup_id, "count": count},
            )
        )
    if duplicates:
        duplicate_counts["categories"] = len(duplicates)

    return diagnostics, duplicate_counts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_coco_annotations(
    annotation_path: Union[str, Path],
    *,
    validate_relationships: bool = True,
) -> CocoParseResult:
    """Load and parse a COCO-style annotation file.

    Args:
        annotation_path: Path to the COCO JSON file.
        validate_relationships: Whether to validate annotation relationships.

    Returns:
        CocoParseResult with normalized records and summary.
    """
    annotation_path = Path(annotation_path)
    data = _read_json(annotation_path)

    result = CocoParseResult()
    summary = CocoSummary()

    # Validate required collections (T016)
    collection_diagnostics = _validate_required_collections(data, annotation_path)
    summary.diagnostics.extend(collection_diagnostics)

    if any(d.severity == "error" for d in collection_diagnostics):
        result.summary = summary
        return result

    # Normalize records (T017)
    result.images = [_normalize_image(img) for img in data.get("images", [])]
    result.categories = [_normalize_category(cat) for cat in data.get("categories", [])]
    result.annotations = [_normalize_annotation(ann) for ann in data.get("annotations", [])]

    summary.image_count = len(result.images)
    summary.annotation_count = len(result.annotations)
    summary.category_count = len(result.categories)

    # Build mappings (T018) — only include annotations for KNOWN images
    known_image_ids = {img.image_id for img in result.images}
    result.image_to_annotations = {}
    for ann in result.annotations:
        if ann.image_id in known_image_ids:
            result.image_to_annotations.setdefault(ann.image_id, []).append(ann)

    result.category_to_annotations = {}
    for ann in result.annotations:
        result.category_to_annotations.setdefault(ann.category_id, []).append(ann)

    summary.images_with_annotations = len(result.image_to_annotations)
    summary.images_without_annotations = summary.image_count - summary.images_with_annotations

    # Validate relationships if requested
    if validate_relationships:
        rel_diagnostics = _validate_relationships(result.images, result.annotations, result.categories)
        summary.diagnostics.extend(rel_diagnostics)
        summary.orphaned_annotations = sum(1 for d in rel_diagnostics if d.code == "orphaned_annotation")
        summary.unknown_categories = sum(1 for d in rel_diagnostics if d.code == "unknown_category")
        summary.malformed_bboxes = sum(1 for d in rel_diagnostics if d.code == "malformed_bbox")

        # Duplicate ID detection (T028)
        dup_diagnostics, dup_counts = _detect_duplicate_ids(result.images, result.annotations, result.categories)
        summary.diagnostics.extend(dup_diagnostics)
        summary.duplicate_ids = dup_counts

    result.summary = summary
    return result
