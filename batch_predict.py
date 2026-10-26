"""Batch inference: score the newest calls with the @champion model and watch for drift.

  python batch_predict.py
  python batch_predict.py --alias challenger --chunks 6

The "new batch" is the most recent 20% of calls, which the model never saw.
It is scored in time order, in --chunks equal slices, and each slice's metrics
are logged as one step so the MLflow UI draws them as a line over time. The
predictions CSV is saved as a run artifact.
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from bankml.data import labels, load_splits
from bankml.model import classification_metrics
from bankml.tracking import CHAMPION, MODEL_NAME, log_dataset, setup

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alias", default=CHAMPION, help="registry alias to score with")
    parser.add_argument("--chunks", type=int, default=4, help="time slices to log metrics for")
    args = parser.parse_args(argv)

    setup()
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, args.alias)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{args.alias}")
    splits = load_splits()
    batch, train = splits["new_batch"], splits["train"]

    with mlflow.start_run(run_name=f"batch-inference-v{version.version}"):
        mlflow.set_tags({"kind": "batch_inference", "model_alias": args.alias,
                         "model_version": version.version})
        log_dataset(batch, "bank-new-batch", "inference")

        out = model.predict(batch)
        y = labels(batch).to_numpy()
        metrics = classification_metrics(y, out["probability"].to_numpy(), out["prediction"].to_numpy())
        mlflow.log_metrics({f"batch_{k}": v for k, v in metrics.items()})
        mlflow.log_metric("train_positive_rate", labels(train).mean())

        for step, idx in enumerate(np.array_split(np.arange(len(batch)), args.chunks)):
            chunk = classification_metrics(y[idx], out["probability"].to_numpy()[idx],
                                           out["prediction"].to_numpy()[idx])
            mlflow.log_metrics({f"chunk_{k}": v for k, v in chunk.items()}, step=step)

        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / "new_batch_predictions.csv"
        pd.concat([batch.reset_index(drop=True), out], axis=1).to_csv(path, index=False)
        mlflow.log_artifact(str(path))

    print(f"Scored {len(batch)} new calls with {MODEL_NAME} v{version.version} (@{args.alias})")
    print(f"  ROC AUC {metrics['roc_auc']:.4f} | PR AUC {metrics['pr_auc']:.4f} | recall {metrics['recall']:.4f}")
    print(f"  actual 'yes' rate {metrics['actual_positive_rate']:.3f} vs "
          f"{labels(train).mean():.3f} in training -> the population has shifted")
    print(f"  predictions saved to {path}")
    return metrics


if __name__ == "__main__":
    main()
