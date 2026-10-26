"""End-to-end MLflow workflow against a throwaway SQLite tracking store."""

import mlflow
import pytest
from mlflow import MlflowClient

import batch_predict
import bankml.data
import promote
import train
from bankml.tracking import CHALLENGER, CHAMPION, EXPERIMENT_NAME, MODEL_NAME
from tests.conftest import make_bank_df


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # artifacts and outputs land in the temp dir
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(bankml.data, "load_dataset", lambda path=None: make_bank_df())
    monkeypatch.setattr(batch_predict, "OUTPUT_DIR", tmp_path / "outputs")
    yield MlflowClient(f"sqlite:///{tmp_path}/mlflow.db")


def aliases(client):
    return client.get_registered_model(MODEL_NAME).aliases


def test_train_promote_serve_cycle(tracking):
    client = tracking

    # Baseline run: logs params, metrics, plots, datasets and a model.
    baseline_id = train.main(["--family", "logreg"])
    run = client.get_run(baseline_id)
    assert run.data.tags["candidate"] == "true"
    assert {"val_roc_auc", "val_pr_auc", "val_recall"} <= run.data.metrics.keys()
    assert {i.dataset.name for i in run.inputs.dataset_inputs} == {"bank-train", "bank-val"}
    artifacts = {a.path for a in client.list_artifacts(baseline_id, "plots")}
    assert "plots/validation.png" in artifacts

    # An impossible gate rejects it: registered as challenger, no champion.
    assert not promote.main(["--run-id", baseline_id, "--min-roc-auc", "0.999"])["promoted"]
    assert aliases(client) == {CHALLENGER: 1}

    # A reachable gate promotes the same version without registering it again.
    assert promote.main(["--run-id", baseline_id, "--min-roc-auc", "0.5", "--min-recall", "0"])["promoted"]
    assert aliases(client) == {CHAMPION: 1}
    assert len(client.search_model_versions(f"name='{MODEL_NAME}'")) == 1

    # Hyperopt search: one nested child run per trial.
    tuned_id = train.main(["--family", "hgb", "--max-evals", "2"])
    children = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME],
                                  filter_string=f"tags.mlflow.parentRunId = '{tuned_id}'")
    assert len(children) == 2

    # The challenger must beat the champion by a margin it cannot reach, so the champion stays.
    result = promote.main(["--run-id", tuned_id, "--min-roc-auc", "0.5", "--min-recall", "0",
                           "--min-improvement", "1.0"])
    assert not result["promoted"]
    assert aliases(client) == {CHAMPION: 1, CHALLENGER: 2}
    assert client.get_model_version(MODEL_NAME, "2").tags["validation_status"] == "failed"

    # Batch inference with the champion logs per-chunk metrics as steps.
    metrics = batch_predict.main(["--chunks", "3"])
    assert 0 <= metrics["roc_auc"] <= 1
    batch_run = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME],
                                   filter_string="tags.kind = 'batch_inference'").iloc[0]
    assert len(client.get_metric_history(batch_run.run_id, "chunk_roc_auc")) == 3
