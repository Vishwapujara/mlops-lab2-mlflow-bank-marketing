"""Train one model family on Bank Marketing and log everything to MLflow.

  python train.py --family rf                    # one baseline run, default parameters
  python train.py --family hgb --max-evals 25    # Hyperopt search with nested runs

A search creates one parent run and one nested child run per trial. The best
parameters are then refit and logged in the parent run as a pyfunc model,
together with validation metrics, plots and feature importances. Parent runs
are tagged candidate=true so promote.py can find them.
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import sklearn
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score, roc_auc_score

from bankml.data import SEED, labels, load_splits, prepare_features
from bankml.model import (DEFAULT_PARAMS, FAMILIES, SEARCH_SPACES, BankMarketingModel,
                          best_threshold, build_pipeline, classification_metrics,
                          evaluation_figure, feature_importance)
from bankml.tracking import log_dataset, setup

CODE_PATH = str(Path(__file__).resolve().parent / "bankml")
PIP_REQUIREMENTS = [
    f"mlflow=={mlflow.__version__}",
    f"scikit-learn=={sklearn.__version__}",
    f"pandas=={pd.__version__}",
    f"numpy=={np.__version__}",
]


def tune(family, max_evals, X_train, y_train, X_val, y_val, seed):
    """Hyperopt TPE search; every trial is logged as a nested MLflow run."""
    space = SEARCH_SPACES[family]

    def objective(params):
        with mlflow.start_run(run_name=f"{family}-trial", nested=True):
            proba = build_pipeline(family, params, seed).fit(X_train, y_train).predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            mlflow.log_params(params)
            mlflow.log_metrics({"val_roc_auc": auc, "val_pr_auc": average_precision_score(y_val, proba)})
        return {"loss": -auc, "status": STATUS_OK}

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials,
                rstate=np.random.default_rng(seed), show_progressbar=False)
    # Log how the search converged: best validation AUC seen after each trial.
    for step, loss in enumerate(np.minimum.accumulate(trials.losses())):
        mlflow.log_metric("best_val_roc_auc_so_far", -loss, step=step)
    return space_eval(space, best)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=FAMILIES, default="rf")
    parser.add_argument("--max-evals", type=int, default=0, help="Hyperopt trials (0 = untuned baseline)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    setup()
    splits = load_splits()
    X_train, y_train = prepare_features(splits["train"]), labels(splits["train"])
    X_val, y_val = prepare_features(splits["val"]), labels(splits["val"])

    kind = "tuned" if args.max_evals else "baseline"
    with mlflow.start_run(run_name=f"{args.family}-{kind}") as run:
        mlflow.set_tags({"family": args.family, "kind": kind, "candidate": "true"})
        log_dataset(splits["train"], "bank-train", "training")
        log_dataset(splits["val"], "bank-val", "validation")

        if args.max_evals:
            params = tune(args.family, args.max_evals, X_train, y_train, X_val, y_val, args.seed)
        else:
            params = DEFAULT_PARAMS[args.family]
        mlflow.log_params({**params, "max_evals": args.max_evals, "seed": args.seed})

        pipeline = build_pipeline(args.family, params, args.seed).fit(X_train, y_train)
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        threshold = best_threshold(y_val, val_proba)
        metrics = classification_metrics(y_val, val_proba, (val_proba >= threshold).astype(int))
        mlflow.log_param("threshold", round(threshold, 4))
        mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

        mlflow.log_figure(evaluation_figure(y_val, val_proba, threshold, f"{run.info.run_name} - validation"),
                          "plots/validation.png")
        mlflow.log_table(feature_importance(pipeline, X_val, y_val), "feature_importance.json")

        model = BankMarketingModel(pipeline, threshold)
        example = X_train.head(5)
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=model,
            code_paths=[CODE_PATH],
            signature=infer_signature(example, model.predict(None, example)),
            input_example=example,
            pip_requirements=PIP_REQUIREMENTS,
        )
        mlflow.set_tag("model_uri", info.model_uri)

    print(f"\nRun {run.info.run_id} ({run.info.run_name})")
    print(f"  params:    {params}")
    print(f"  threshold: {threshold:.3f}")
    for k in ["roc_auc", "pr_auc", "f1", "precision", "recall"]:
        print(f"  val_{k:<9} {metrics[k]:.4f}")
    return run.info.run_id


if __name__ == "__main__":
    main()
