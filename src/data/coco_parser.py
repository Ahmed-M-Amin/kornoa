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
        # Malformed bbox
        if ann.bbox is not None:
            x, y, w, h = ann.bbox
            if any(v is None for v in (x, y, w, h)) or x < 0 or y < 0 or w <= 0 or h <= 0:
                diagnostics.append(
                    CocoDiagnostic(
                        severity="warning",
                        code="malformed_bbox",
                        message=f"Annotation {ann.annotation_id} has invalid bbox {ann.bbox}",
                        context={"annotation_id": ann.annotation_id, "bbox": ann.bbox},
                    )
                )

    return diagnostics


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

    # Build mappings (T018)
    result.image_to_annotations = {}
    for ann in result.annotations:
        result.image_to_annotations.setdefault(ann.image_id, []).append(ann)

    result.category_to_annotations = {}
    for ann in result.annotations:
        result.category_to_annotations.setdefault(ann.category_id, []).append(ann)

    known_image_ids = {image.image_id for image in result.images}
    known_images_with_annotations = {
        image_id
        for image_id in known_image_ids
        if result.image_to_annotations.get(image_id)
    }
    summary.images_with_annotations = len(known_images_with_annotations)
    summary.images_without_annotations = summary.image_count - summary.images_with_annotations

    # Validate relationships if requested
    if validate_relationships:
        rel_diagnostics = _validate_relationships(result.images, result.annotations, result.categories)
        summary.diagnostics.extend(rel_diagnostics)
        summary.orphaned_annotations = sum(1 for d in rel_diagnostics if d.code == "orphaned_annotation")
        summary.unknown_categories = sum(1 for d in rel_diagnostics if d.code == "unknown_category")
        summary.malformed_bboxes = sum(1 for d in rel_diagnostics if d.code == "malformed_bbox")

    result.summary = summary
    return result
