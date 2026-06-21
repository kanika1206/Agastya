import pytest

from agastya.verify.calibration import (
    Calibrator,
    conformal_quantile,
    conformal_set,
    expected_calibration_error,
    fit_calibrator,
    fit_temperature,
    needs_human_review,
    temperature_scale,
)


def test_temperature_one_is_identity():
    assert temperature_scale(0.8, 1.0) == pytest.approx(0.8)


def test_temperature_above_one_softens_toward_half():
    assert temperature_scale(0.9, 2.0) < 0.9
    assert temperature_scale(0.9, 2.0) > 0.5


def test_temperature_below_one_sharpens_away_from_half():
    assert temperature_scale(0.7, 0.5) > 0.7


def test_temperature_large_collapses_to_half():
    assert temperature_scale(0.99, 100000.0) == pytest.approx(0.5, abs=1e-3)


def test_temperature_rejects_nonpositive():
    with pytest.raises(ValueError):
        temperature_scale(0.8, 0.0)


def test_fit_temperature_softens_overconfident_model():
    probs = [0.99] * 10
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    temperature = fit_temperature(probs, labels)
    assert temperature > 1.0


def test_fit_temperature_sharpens_underconfident_model():
    probs = [0.55] * 5 + [0.45] * 5
    labels = [1] * 5 + [0] * 5
    temperature = fit_temperature(probs, labels)
    assert temperature < 1.0


def test_ece_zero_for_perfectly_calibrated():
    probs = [0.0, 0.0, 1.0, 1.0]
    labels = [0, 0, 1, 1]
    assert expected_calibration_error(probs, labels, bins=2) == pytest.approx(0.0)


def test_ece_positive_for_overconfident():
    probs = [0.9, 0.9, 0.9, 0.9]
    labels = [1, 0, 0, 0]
    assert expected_calibration_error(probs, labels, bins=1) == pytest.approx(0.65)


def test_conformal_quantile_finite_sample_corrected():
    scores = [0.1, 0.2, 0.3, 0.4]
    assert conformal_quantile(scores, alpha=0.2) == pytest.approx(0.4)


def test_conformal_quantile_clamps_to_one_when_insufficient_data():
    assert conformal_quantile([0.5], alpha=0.05) == 1.0


def test_conformal_set_includes_plausible_labels():
    probs = {"violation": 0.8, "clean": 0.2}
    assert conformal_set(probs, qhat=0.5) == {"violation"}


def test_conformal_set_can_be_ambiguous():
    probs = {"violation": 0.55, "clean": 0.45}
    assert conformal_set(probs, qhat=0.6) == {"violation", "clean"}


def test_needs_review_when_set_not_singleton():
    assert needs_human_review({"violation", "clean"}) is True
    assert needs_human_review({"violation"}) is False
    assert needs_human_review(set()) is True


def test_calibrator_scales_confidence_by_temperature():
    cal = Calibrator(temperature=2.0, qhat=0.3)
    result = cal.evaluate(0.9, "triple-riding")
    assert result.confidence == pytest.approx(temperature_scale(0.9, 2.0))


def test_calibrator_confident_case_is_singleton_no_review():
    cal = Calibrator(temperature=2.0, qhat=0.3)
    result = cal.evaluate(0.9, "triple-riding")
    assert result.prediction_set == {"triple-riding"}
    assert result.needs_review is False


def test_calibrator_borderline_case_is_ambiguous_needs_review():
    cal = Calibrator(temperature=2.0, qhat=0.55)
    result = cal.evaluate(0.55, "no-helmet")
    assert result.prediction_set == {"no-helmet", "clean"}
    assert result.needs_review is True


def test_calibrator_rejects_invalid_qhat():
    with pytest.raises(ValueError):
        Calibrator(temperature=1.0, qhat=-0.1)


def test_fit_calibrator_returns_calibrator_with_valid_qhat():
    confidences = [0.9, 0.8, 0.7, 0.6, 0.55, 0.5]
    labels = [1, 1, 0, 1, 0, 0]
    cal = fit_calibrator(confidences, labels, alpha=0.1)
    assert isinstance(cal, Calibrator)
    assert 0.0 <= cal.qhat <= 1.0


def test_fit_calibrator_softens_overconfident_model():
    confidences = [0.99] * 10
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    cal = fit_calibrator(confidences, labels, alpha=0.1)
    assert cal.temperature > 1.0
