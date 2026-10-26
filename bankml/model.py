"""Model families, search spaces, metrics and the MLflow pyfunc wrapper."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow.pyfunc
import numpy as np
import pandas as pd
from hyperopt import hp
from hyperopt.pyll import scope
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bankml.data import CATEGORICAL, NUMERIC, SEED, prepare_features

FAMILIES = ["logreg", "rf", "hgb"]

# Untuned defaults. rf matches the original lab's baseline (10 trees).
DEFAULT_PARAMS = {
    "logreg": {"C": 1.0, "class_weight": None},
    "rf": {"n_estimators": 10, "max_depth": None, "min_samples_leaf": 1, "class_weight": None},
    "hgb": {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31,
            "min_samples_leaf": 20, "l2_regularization": 0.0},
}

SEARCH_SPACES = {
    "logreg": {
        "C": hp.loguniform("C", math.log(1e-3), math.log(10)),
        "class_weight": hp.choice("class_weight", [None, "balanced"]),
    },
    "rf": {
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 400, 50)),
        "max_depth": scope.int(hp.quniform("max_depth", 4, 20, 2)),
        "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 20, 1)),
        "class_weight": hp.choice("class_weight", [None, "balanced_subsample"]),
    },
    "hgb": {
        "learning_rate": hp.loguniform("learning_rate", math.log(0.01), math.log(0.3)),
        "max_iter": scope.int(hp.quniform("max_iter", 100, 500, 50)),
        "max_leaf_nodes": scope.int(hp.quniform("max_leaf_nodes", 8, 64, 4)),
        "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 10, 100, 10)),
        "l2_regularization": hp.loguniform("l2_regularization", math.log(1e-4), math.log(10)),
    },
}


def build_pipeline(family, params, seed=SEED):
    """Preprocessing + classifier as one sklearn Pipeline that takes raw feature columns."""
    if family == "logreg":
        clf = LogisticRegression(C=params["C"], class_weight=params["class_weight"], max_iter=2000)
    elif family == "rf":
        clf = RandomForestClassifier(n_jobs=-1, random_state=seed, **params)
    elif family == "hgb":
        clf = HistGradientBoostingClassifier(random_state=seed, **params)
    else:
        raise ValueError(f"Unknown model family {family!r}, expected one of {FAMILIES}")
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def best_threshold(y_true, proba):
    """Decision threshold that maximises F1 (the classes are imbalanced, so 0.5 is a poor default)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def classification_metrics(y_true, proba, pred):
    """Ranking metrics from the probabilities, decision metrics from the 0/1 predictions."""
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
        "predicted_positive_rate": float(pred.mean()),
        "actual_positive_rate": float(np.mean(y_true)),
    }


def evaluation_figure(y_true, proba, threshold, title):
    """ROC curve, precision-recall curve and confusion matrix side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, proba):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    axes[0].legend()
    precision, recall, _ = precision_recall_curve(y_true, proba)
    axes[1].plot(recall, precision, label=f"AP = {average_precision_score(y_true, proba):.3f}")
    axes[1].axhline(np.mean(y_true), ls="--", color="grey", label="random")
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    axes[1].legend()
    ConfusionMatrixDisplay.from_predictions(y_true, (proba >= threshold).astype(int),
                                            display_labels=["no", "yes"], ax=axes[2], colorbar=False)
    axes[2].set_title(f"Confusion matrix (threshold {threshold:.2f})")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def feature_importance(pipeline, X, y):
    """Importance per input column (one-hot columns summed back to their source column)."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_[0])
    else:
        # HistGradientBoosting has no built-in importances, so use permutation importance.
        from sklearn.inspection import permutation_importance
        result = permutation_importance(pipeline, X, y, scoring="roc_auc", n_repeats=3,
                                        random_state=SEED, n_jobs=-1)
        return (pd.DataFrame({"feature": X.columns, "importance": result.importances_mean})
                .sort_values("importance", ascending=False, ignore_index=True))
    # Transformed columns are the numeric ones, then one block of one-hot columns per category.
    categories = pipeline.named_steps["pre"].named_transformers_["cat"].categories_
    source = NUMERIC + [col for col, cats in zip(CATEGORICAL, categories) for _ in cats]
    return (pd.DataFrame({"feature": source, "importance": values})
            .groupby("feature", as_index=False)["importance"].sum()
            .sort_values("importance", ascending=False, ignore_index=True))


class BankMarketingModel(mlflow.pyfunc.PythonModel):
    """sklearn pipeline plus its tuned decision threshold.

    Takes raw customer rows (missing categories allowed) and returns both the
    probability of subscribing and the yes/no decision at the tuned threshold.
    """

    def __init__(self, pipeline, threshold):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict(self, context, model_input, params=None):
        proba = self.pipeline.predict_proba(prepare_features(model_input))[:, 1]
        return pd.DataFrame({"probability": proba,
                             "prediction": (proba >= self.threshold).astype(int)})
