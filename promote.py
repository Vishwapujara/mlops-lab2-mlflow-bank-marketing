"""Register the best candidate run and promote it to @champion only if it passes the gates.

  python promote.py                 # best candidate by validation ROC AUC
  python promote.py --run-id <id>   # a specific training run

Steps:
  1. Register the run's model as a new version of bank-marketing-classifier
     and point the @challenger alias at it.
  2. Score @challenger and the current @champion on the held-out test set.
  3. Gates: challenger ROC AUC >= --min-roc-auc, recall >= --min-recall, and
     (if a champion exists) ROC AUC at least --min-improvement above the champion.
  4. Pass: move @champion to the new version. Fail: keep the old champion.

Every promotion attempt is logged as its own MLflow run with both models'
test metrics, a side-by-side ROC plot and the gate results.
"""

import argparse
import json

import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import roc_curve

from bankml.data import FEATURES, labels, load_splits
from bankml.model import classification_metrics
from bankml.tracking import CHALLENGER, CHAMPION, EXPERIMENT_NAME, MODEL_NAME, log_dataset, setup


def best_candidate():
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="tags.candidate = 'true' and attributes.status = 'FINISHED'",
        order_by=["metrics.val_roc_auc DESC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit("No finished training runs found. Run train.py first.")
    return runs.iloc[0].run_id


def register(client, run_id):
    """Return the model version for this run, registering it if it is new."""
    existing = client.search_model_versions(f"name='{MODEL_NAME}' and run_id='{run_id}'")
    if existing:
        return existing[0], False
    run = client.get_run(run_id)
    version = mlflow.register_model(run.data.tags["model_uri"], MODEL_NAME,
                                    tags={"family": run.data.tags["family"], "kind": run.data.tags["kind"]})
    return version, True


def current_champion(client):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
    except MlflowException:
        return None


def score(alias, X, y):
    """Load a model by registry alias and compute test metrics from what it would serve."""
    out = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}").predict(X)
    proba = out["probability"].to_numpy()
    return proba, classification_metrics(y, proba, out["prediction"].to_numpy())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", help="training run to register (default: best by val_roc_auc)")
    parser.add_argument("--min-roc-auc", type=float, default=0.68)
    parser.add_argument("--min-recall", type=float, default=0.25)
    parser.add_argument("--min-improvement", type=float, default=0.002,
                        help="ROC AUC the challenger must gain over the champion")
    args = parser.parse_args(argv)

    setup()
    client = MlflowClient()
    run_id = args.run_id or best_candidate()
    version, is_new = register(client, run_id)
    champion = current_champion(client)
    print(f"Candidate run {run_id} -> {MODEL_NAME} v{version.version}"
          f" ({'newly registered' if is_new else 'already registered'})")

    if champion and champion.version == version.version:
        print(f"v{version.version} is already @{CHAMPION}. Train a better model first.")
        return {"promoted": False, "version": version.version, "reason": "already champion"}

    client.set_registered_model_alias(MODEL_NAME, CHALLENGER, version.version)
    test = load_splits()["test"]
    y_test = labels(test)

    with mlflow.start_run(run_name=f"promote-v{version.version}"):
        mlflow.set_tags({"kind": "promotion", "challenger_version": version.version,
                         "champion_version": champion.version if champion else "none"})
        log_dataset(test, "bank-test", "test")

        ch_proba, ch = score(CHALLENGER, test[FEATURES], y_test)
        mlflow.log_metrics({f"challenger_test_{k}": v for k, v in ch.items()})
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(*roc_curve(y_test, ch_proba)[:2], label=f"challenger v{version.version} ({ch['roc_auc']:.3f})")

        gates = {
            "min_roc_auc": {"passed": ch["roc_auc"] >= args.min_roc_auc,
                            "value": ch["roc_auc"], "required": args.min_roc_auc},
            "min_recall": {"passed": ch["recall"] >= args.min_recall,
                           "value": ch["recall"], "required": args.min_recall},
        }
        if champion:
            cp_proba, cp = score(CHAMPION, test[FEATURES], y_test)
            mlflow.log_metrics({f"champion_test_{k}": v for k, v in cp.items()})
            ax.plot(*roc_curve(y_test, cp_proba)[:2], label=f"champion v{champion.version} ({cp['roc_auc']:.3f})")
            gates["beats_champion"] = {"passed": ch["roc_auc"] >= cp["roc_auc"] + args.min_improvement,
                                       "value": ch["roc_auc"] - cp["roc_auc"],
                                       "required": args.min_improvement}

        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set(title="Test ROC: challenger vs champion", xlabel="False positive rate", ylabel="True positive rate")
        ax.legend()
        mlflow.log_figure(fig, "plots/challenger_vs_champion.png")
        plt.close(fig)

        promoted = all(g["passed"] for g in gates.values())
        mlflow.log_params({f"gate_{k}": v["required"] for k, v in gates.items()})
        mlflow.log_dict(gates, "gate_results.json")
        mlflow.log_metric("promoted", int(promoted))

    client.set_model_version_tag(MODEL_NAME, version.version, "validation_status",
                                 "passed" if promoted else "failed")
    client.set_model_version_tag(MODEL_NAME, version.version, "test_roc_auc", f"{ch['roc_auc']:.4f}")
    if promoted:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, version.version)
        client.delete_registered_model_alias(MODEL_NAME, CHALLENGER)
        if champion:
            client.set_model_version_tag(MODEL_NAME, champion.version, "replaced_by", version.version)

    print(json.dumps(gates, indent=2, default=float))
    if promoted:
        print(f"PROMOTED: v{version.version} is now @{CHAMPION}"
              + (f" (replaced v{champion.version})" if champion else ""))
    else:
        print(f"REJECTED: v{version.version} stays @{CHALLENGER}; "
              + (f"@{CHAMPION} is still v{champion.version}" if champion else "no champion yet"))
    return {"promoted": promoted, "version": version.version}


if __name__ == "__main__":
    main()
