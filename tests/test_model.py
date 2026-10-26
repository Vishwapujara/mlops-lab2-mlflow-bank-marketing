import numpy as np
import pytest

from bankml.data import labels, prepare_features
from bankml.model import (DEFAULT_PARAMS, FAMILIES, BankMarketingModel, best_threshold,
                          build_pipeline, classification_metrics, feature_importance)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_trains_and_beats_random(bank_df, family):
    X, y = prepare_features(bank_df), labels(bank_df)
    proba = build_pipeline(family, DEFAULT_PARAMS[family]).fit(X, y).predict_proba(X)[:, 1]
    assert classification_metrics(y, proba, (proba > 0.5).astype(int))["roc_auc"] > 0.6


@pytest.mark.parametrize("family", FAMILIES)
def test_feature_importance_covers_every_input_column(bank_df, family):
    X, y = prepare_features(bank_df), labels(bank_df)
    pipeline = build_pipeline(family, DEFAULT_PARAMS[family]).fit(X, y)
    importance = feature_importance(pipeline, X, y)
    assert sorted(importance["feature"]) == sorted(X.columns)


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError):
        build_pipeline("svm", {})


def test_best_threshold_finds_the_separating_cutoff():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    assert best_threshold(y, proba) == pytest.approx(0.7)


def test_pyfunc_accepts_raw_rows_with_missing_values(bank_df):
    X, y = prepare_features(bank_df), labels(bank_df)
    model = BankMarketingModel(build_pipeline("logreg", DEFAULT_PARAMS["logreg"]).fit(X, y), threshold=0.3)
    raw = bank_df.head(20)
    assert raw[["job", "contact", "poutcome"]].isna().any().any()
    out = model.predict(None, raw)
    assert list(out.columns) == ["probability", "prediction"]
    assert out["probability"].between(0, 1).all()
    assert (out["prediction"] == (out["probability"] >= 0.3)).all()
