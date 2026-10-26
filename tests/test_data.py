import numpy as np
import pytest

from bankml.data import CATEGORICAL, FEATURES, NEW_BATCH_FRACTION, labels, prepare_features, split


def test_prepare_features_drops_leaky_and_target_columns(bank_df):
    X = prepare_features(bank_df)
    assert list(X.columns) == FEATURES
    assert "duration" not in X and "y" not in X


def test_prepare_features_fills_missing_categories(bank_df):
    X = prepare_features(bank_df)
    assert not X[CATEGORICAL].isna().any().any()
    assert (X["poutcome"] == "unknown").sum() == bank_df["poutcome"].isna().sum()


def test_prepare_features_rejects_missing_columns(bank_df):
    with pytest.raises(ValueError, match="age"):
        prepare_features(bank_df.drop(columns=["age"]))


def test_labels_are_binary(bank_df):
    y = labels(bank_df)
    assert set(y.unique()) <= {0, 1}
    assert y.sum() == (bank_df["y"] == "yes").sum()


def test_split_sizes_and_no_overlap(bank_df):
    parts = split(bank_df)
    assert sum(len(p) for p in parts.values()) == len(bank_df)
    indexes = [set(p.index) for p in parts.values()]
    assert all(a.isdisjoint(b) for i, a in enumerate(indexes) for b in indexes[i + 1:])
    assert len(parts["new_batch"]) == len(bank_df) - int(len(bank_df) * (1 - NEW_BATCH_FRACTION))


def test_new_batch_is_the_most_recent_rows(bank_df):
    parts = split(bank_df)
    assert parts["new_batch"].index.min() > max(parts[k].index.max() for k in ["train", "val", "test"])


def test_split_is_stratified(bank_df):
    parts = split(bank_df)
    rates = [labels(parts[k]).mean() for k in ["train", "val", "test"]]
    assert np.ptp(rates) < 0.01
