"""MLflow tracking-server connection and names shared by all scripts."""

import os
import sys

import mlflow
import requests

from bankml.data import DATA_URL

DEFAULT_TRACKING_URI = "http://127.0.0.1:5050"
EXPERIMENT_NAME = "bank-marketing"
MODEL_NAME = "bank-marketing-classifier"
CHAMPION = "champion"
CHALLENGER = "challenger"


def setup():
    """Point MLflow at the tracking server (MLFLOW_TRACKING_URI, default localhost:5050)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    if uri.startswith("http"):
        try:
            requests.get(f"{uri}/health", timeout=5).raise_for_status()
        except requests.RequestException:
            sys.exit(f"Cannot reach the MLflow tracking server at {uri}.\n"
                     "Start it first (see README, step 3) or set MLFLOW_TRACKING_URI.")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri


def log_dataset(df, name, context):
    """Record which data a run used, so it shows up in the run's Datasets panel."""
    dataset = mlflow.data.from_pandas(df, source=DATA_URL, targets="y", name=name)
    mlflow.log_input(dataset, context=context)

