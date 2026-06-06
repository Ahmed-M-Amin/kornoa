"""Tests for SPEC-009 hybrid fusion behavior."""

from __future__ import annotations

import json

import pandas as pd
import pytest


def _classifier_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image": ["images/a.jpg", "images/b.jpg", "images/c.jpg"],
            "probability": [0.9, 0.1, 0.52],
            "label": [1, 0, 1],
        }
    )


def _detector_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filename": ["a.jpg", "a.jpg", "c.jpg"],
            "confidence": [0.2, 0.8, 0.7],
            "category": ["scratch", "contamination", "crack"],
            "area": [0.01, 0.05, 0.02],
        }
    )


def test_classifier_schema_normalization_and_threshold_targets(tmp_path):
    from src.inference.fusion import load_best_threshold, normalize_classifier_predictions

    threshold_path = tmp_path / "best_threshold.json"
    threshold_path.write_text(json.dumps({"threshold": 0.55}), encoding="utf-8")

    normalized = normalize_classifier_predictions(_classifier_frame(), load_best_threshold(threshold_path), require_label=True)

    assert normalized["image_id"].tolist() == ["a.jpg", "b.jpg", "c.jpg"]
    assert normalized["classifier_score"].tolist() == pytest.approx([0.9, 0.1, 0.52])
    assert normalized["classifier_target"].tolist() == [1, 0, 0]
    assert normalized["y_true"].tolist() == [1, 0, 1]


def test_detector_schema_normalization_preserves_multi_row_candidates():
    from src.inference.fusion import normalize_detector_predictions

    normalized = normalize_detector_predictions(_detector_frame())

    assert normalized["image_id"].tolist() == ["a.jpg", "a.jpg", "c.jpg"]
    assert normalized["detector_confidence"].tolist() == pytest.approx([0.8, 0.2, 0.7])
    assert normalized["defect_category"].tolist() == ["contamination", "scratch", "crack"]


def test_detector_aggregation_keeps_lower_confidence_rejection_candidate():
    from src.inference.fusion import HybridFusionConfig, apply_hybrid_fusion

    classifier = pd.DataFrame({"image_id": ["a.jpg"], "classifier_score": [0.49], "classifier_target": [0]})
    detector = pd.DataFrame(
        {
            "image_id": ["a.jpg", "a.jpg"],
            "detector_confidence": [0.95, 0.70],
            "defect_category": ["no_fault", "crack"],
            "defect_area": [0.0, 0.01],
        }
    )

    fused = apply_hybrid_fusion(
        classifier,
        detector,
        HybridFusionConfig(0.5, 0.05, 0.25, always_faulty_categories={"crack"}),
    )

    assert fused.loc[0, "hybrid_target"] == 1
    assert fused.loc[0, "detector_used"] is True
    assert fused.loc[0, "detector_rejection_reason"] == "always_faulty:crack"


def test_detector_normalization_accepts_real_binary_detector_columns():
    from src.inference.fusion import normalize_detector_predictions

    raw = pd.DataFrame(
        {
            "image_id": ["a.jpg"],
            "max_detector_conf": [0.91],
            "prediction": [1],
            "n_boxes": [3],
        }
    )

    normalized = normalize_detector_predictions(raw)

    assert normalized.loc[0, "detector_confidence"] == pytest.approx(0.91)
    assert normalized.loc[0, "detector_binary_target"] == 1


def test_binary_metrics_include_zero_division_cases():
    from src.inference.fusion import compute_binary_metrics

    metrics = compute_binary_metrics([0, 0], [0, 0])

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)
    assert metrics["tn"] == 2
    assert metrics["support_positive"] == 0
    assert metrics["support_negative"] == 2


def test_confident_classifier_predictions_remain_unchanged():
    from src.inference.fusion import HybridFusionConfig, apply_hybrid_fusion

    classifier = pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg"],
            "classifier_score": [0.95, 0.05],
            "classifier_target": [1, 0],
        }
    )
    detector = pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg"],
            "detector_confidence": [0.99, 0.99],
            "defect_category": ["crack", "crack"],
            "defect_area": [0.2, 0.2],
        }
    )

    fused = apply_hybrid_fusion(classifier, detector, HybridFusionConfig(0.5, 0.1, 0.25, always_faulty_categories={"crack"}))

    assert fused["hybrid_target"].tolist() == [1, 0]
    assert fused["detector_used"].tolist() == [False, False]
    assert fused["decision_source"].tolist() == ["classifier", "classifier"]


def test_detector_disagreement_cannot_override_confident_rows():
    from src.inference.fusion import HybridFusionConfig, apply_hybrid_fusion

    classifier = pd.DataFrame({"image_id": ["a.jpg"], "classifier_score": [0.9], "classifier_target": [1]})
    detector = pd.DataFrame(
        {"image_id": ["a.jpg"], "detector_confidence": [0.01], "defect_category": ["none"], "defect_area": [0.0]}
    )

    fused = apply_hybrid_fusion(classifier, detector, HybridFusionConfig(0.5, 0.1, 0.25))

    assert fused.loc[0, "hybrid_target"] == 1
    assert fused.loc[0, "detector_used"] is False


def test_detector_usage_report_counts_only_uncertain_eligible_rows():
    from src.inference.fusion import HybridFusionConfig, apply_hybrid_fusion, summarize_detector_usage

    classifier = pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg", "c.jpg"],
            "classifier_score": [0.95, 0.52, 0.48],
            "classifier_target": [1, 1, 0],
            "y_true": [1, 1, 0],
        }
    )
    detector = pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg"],
            "detector_confidence": [0.99, 0.99],
            "defect_category": ["crack", "crack"],
            "defect_area": [0.2, 0.2],
        }
    )
    fused = apply_hybrid_fusion(classifier, detector, HybridFusionConfig(0.5, 0.05, 0.25, always_faulty_categories={"crack"}))

    usage = summarize_detector_usage(fused)

    assert usage["count"] == 1
    assert usage["ratio"] == pytest.approx(1 / 3)


def test_always_faulty_defect_rejects_uncertain_rows():
    from src.inference.fusion import HybridFusionConfig, apply_hybrid_fusion

    classifier = pd.DataFrame({"image_id": ["a.jpg"], "classifier_score": [0.49], "classifier_target": [0]})
    detector = pd.DataFrame(
        {"image_id": ["a.jpg"], "detector_confidence": [0.9], "defect_category": ["crack"], "defect_area": [0.01]}
    )

    fused = apply_hybrid_fusion(classifier, detector, HybridFusionConfig(0.5, 0.05, 0.25, always_faulty_categories={"crack"}))

    assert fused.loc[0, "hybrid_target"] == 1
    assert fused.loc[0, "detector_used"] is True
    assert fused.loc[0, "detector_rejection_reason"] == "always_faulty:crack"


def test_conditional_defect_uses_per_category_area_threshold():
    from src.inference.fusion import HybridFusionConfig, apply_hybrid_fusion

    classifier = pd.DataFrame({"image_id": ["a.jpg", "b.jpg"], "classifier_score": [0.49, 0.49], "classifier_target": [0, 0]})
    detector = pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg"],
            "detector_confidence": [0.9, 0.9],
            "defect_category": ["scuff", "scuff"],
            "defect_area": [0.06, 0.01],
        }
    )

    fused = apply_hybrid_fusion(
        classifier,
        detector,
        HybridFusionConfig(
            0.5,
            0.05,
            0.25,
            conditional_area_thresholds={"scuff": 0.05},
            conditional_categories={"scuff"},
        ),
    )

    assert fused["hybrid_target"].tolist() == [1, 0]
    assert fused["detector_used"].tolist() == [True, False]


def test_default_conditional_area_threshold_fallback_is_reported():
    from src.inference.fusion import HybridFusionConfig, apply_hybrid_fusion, summarize_detector_usage

    classifier = pd.DataFrame({"image_id": ["a.jpg"], "classifier_score": [0.49], "classifier_target": [0]})
    detector = pd.DataFrame(
        {"image_id": ["a.jpg"], "detector_confidence": [0.9], "defect_category": ["unknown_conditional"], "defect_area": [0.04]}
    )

    fused = apply_hybrid_fusion(
        classifier,
        detector,
        HybridFusionConfig(
            0.5,
            0.05,
            0.25,
            conditional_categories={"unknown_conditional"},
            default_conditional_area_threshold=0.03,
        ),
    )
    usage = summarize_detector_usage(fused)

    assert fused.loc[0, "hybrid_target"] == 1
    assert fused.loc[0, "used_default_area_threshold"] is True
    assert usage["fallback_area_threshold_uses"] == 1


def test_missing_invalid_or_weak_detector_evidence_falls_back_to_classifier():
    from src.inference.fusion import HybridFusionConfig, apply_hybrid_fusion

    classifier = pd.DataFrame(
        {"image_id": ["missing.jpg", "weak.jpg"], "classifier_score": [0.49, 0.49], "classifier_target": [0, 0]}
    )
    detector = pd.DataFrame(
        {"image_id": ["weak.jpg"], "detector_confidence": [0.1], "defect_category": ["crack"], "defect_area": [0.2]}
    )

    fused = apply_hybrid_fusion(classifier, detector, HybridFusionConfig(0.5, 0.05, 0.25, always_faulty_categories={"crack"}))

    assert fused["hybrid_target"].tolist() == [0, 0]
    assert fused["detector_used"].tolist() == [False, False]


def test_detector_cannot_clear_uncertain_classifier_rejection():
    from src.inference.fusion import HybridFusionConfig, apply_hybrid_fusion

    classifier = pd.DataFrame({"image_id": ["a.jpg"], "classifier_score": [0.52], "classifier_target": [1]})
    detector = pd.DataFrame(
        {"image_id": ["a.jpg"], "detector_confidence": [0.01], "defect_category": ["none"], "defect_area": [0.0]}
    )

    fused = apply_hybrid_fusion(classifier, detector, HybridFusionConfig(0.5, 0.05, 0.25))

    assert fused.loc[0, "hybrid_target"] == 1
    assert fused.loc[0, "detector_used"] is False


def test_overlap_report_excludes_non_overlap_rows_from_search():
    from src.inference.fusion import build_overlap_report

    classifier = pd.DataFrame({"image_id": ["a.jpg", "b.jpg", "c.jpg"], "y_true": [0, 1, 0]})
    detector = pd.DataFrame({"image_id": ["a.jpg", "b.jpg", "d.jpg"], "detector_confidence": [0.1, 0.2, 0.3]})

    report = build_overlap_report(classifier, detector)

    assert report["overlap_count"] == 2
    assert report["classifier_only_count"] == 1
    assert report["detector_only_count"] == 1
    assert report["overlap_image_ids"] == ["a.jpg", "b.jpg"]


def test_parameter_search_prefers_usage_under_cap_for_close_f1():
    from src.inference.fusion import search_hybrid_parameters

    classifier = pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "classifier_score": [0.49, 0.51, 0.9, 0.1],
            "classifier_target": [0, 1, 1, 0],
            "y_true": [1, 1, 1, 0],
        }
    )
    detector = pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "detector_confidence": [0.9, 0.9, 0.9, 0.0],
            "defect_category": ["crack", "crack", "crack", "none"],
            "defect_area": [0.2, 0.2, 0.2, 0.0],
        }
    )

    config, metrics, _ = search_hybrid_parameters(
        classifier,
        detector,
        base_classifier_threshold=0.5,
        uncertainty_margin_candidates=[0.02, 0.5],
        detector_conf_threshold_candidates=[0.25],
        always_faulty_categories={"crack"},
        detector_usage_preference=0.3,
    )

    assert config.uncertainty_margin == pytest.approx(0.02)
    assert metrics["detector_usage"]["ratio"] <= 0.3


def test_parameter_search_applies_close_f1_tolerance_before_usage():
    from src.inference.fusion import search_hybrid_parameters

    classifier = pd.DataFrame(
        {
            "image_id": [f"img_{idx}.jpg" for idx in range(1000)],
            "classifier_score": [0.49] * 200 + [0.9] * 800,
            "classifier_target": [0] * 200 + [1] * 800,
            "y_true": [1] * 200 + [1] * 800,
        }
    )
    detector = pd.DataFrame(
        {
            "image_id": [f"img_{idx}.jpg" for idx in range(1000)],
            "detector_confidence": [0.9] * 200 + [0.9] * 800,
            "defect_category": ["crack"] * 200 + ["no_fault"] * 800,
            "defect_area": [0.1] * 1000,
        }
    )

    config, metrics, _ = search_hybrid_parameters(
        classifier,
        detector,
        base_classifier_threshold=0.5,
        uncertainty_margin_candidates=[0.02, 0.5],
        detector_conf_threshold_candidates=[0.25],
        always_faulty_categories={"crack"},
        detector_usage_preference=0.30,
    )

    assert config.uncertainty_margin == pytest.approx(0.02)
    assert metrics["hybrid"]["f1"] >= 0.998
    assert metrics["detector_usage"]["ratio"] <= 0.30


def test_parameter_search_evaluates_area_threshold_candidates():
    from src.inference.fusion import search_hybrid_parameters

    classifier = pd.DataFrame(
        {"image_id": ["a.jpg", "b.jpg"], "classifier_score": [0.49, 0.49], "classifier_target": [0, 0], "y_true": [1, 0]}
    )
    detector = pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg"],
            "detector_confidence": [0.9, 0.9],
            "defect_category": ["scuff", "scuff"],
            "defect_area": [0.06, 0.03],
        }
    )

    config, metrics, _ = search_hybrid_parameters(
        classifier,
        detector,
        base_classifier_threshold=0.5,
        uncertainty_margin_candidates=[0.05],
        detector_conf_threshold_candidates=[0.25],
        conditional_categories={"scuff"},
        conditional_area_threshold_candidates={"scuff": [0.02, 0.05]},
        default_conditional_area_threshold_candidates=[0.05],
    )

    assert config.conditional_area_thresholds["scuff"] == pytest.approx(0.05)
    assert metrics["hybrid"]["f1"] == pytest.approx(1.0)


def test_search_reports_baselines_on_same_overlap_set():
    from src.inference.fusion import search_hybrid_parameters

    classifier = pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg", "c.jpg"],
            "classifier_score": [0.49, 0.9, 0.1],
            "classifier_target": [0, 1, 0],
            "y_true": [1, 1, 0],
        }
    )
    detector = pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg"],
            "detector_confidence": [0.9, 0.0],
            "defect_category": ["crack", "none"],
            "defect_area": [0.2, 0.0],
        }
    )

    _, metrics, predictions = search_hybrid_parameters(
        classifier,
        detector,
        base_classifier_threshold=0.5,
        uncertainty_margin_candidates=[0.05],
        detector_conf_threshold_candidates=[0.25],
        always_faulty_categories={"crack"},
    )

    assert metrics["overlap_count"] == 2
    assert metrics["classifier_baseline"]["support_positive"] == 2
    assert metrics["detector_baseline"]["support_positive"] == 2
    assert metrics["hybrid"]["support_positive"] == 2
    assert predictions["image_id"].tolist() == ["a.jpg", "b.jpg"]


def test_detector_baseline_uses_rejection_rules_not_confidence_only():
    from src.inference.fusion import search_hybrid_parameters

    classifier = pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg"],
            "classifier_score": [0.49, 0.49],
            "classifier_target": [0, 0],
            "y_true": [0, 1],
        }
    )
    detector = pd.DataFrame(
        {
            "image_id": ["a.jpg", "b.jpg"],
            "detector_confidence": [0.95, 0.70],
            "defect_category": ["no_fault", "crack"],
            "defect_area": [0.0, 0.1],
        }
    )

    _, metrics, _ = search_hybrid_parameters(
        classifier,
        detector,
        base_classifier_threshold=0.5,
        uncertainty_margin_candidates=[0.05],
        detector_conf_threshold_candidates=[0.25],
        always_faulty_categories={"crack"},
    )

    assert metrics["detector_baseline"]["fp"] == 0
    assert metrics["detector_baseline"]["tp"] == 1
