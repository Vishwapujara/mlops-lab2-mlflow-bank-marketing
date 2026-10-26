# MLOps Lab – MLflow Experiment Tracking, Model Registry & Serving

Lab submission for DADS7305 MLOps (Northeastern University), Experiment Tracking topic.
Based on the course lab [Experiment_Tracking_Labs/Mlflow_Labs/Lab2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab2)
(Wine Quality Prediction).

The original lab trains a random forest on wine data, logs one parameter and one metric,
registers the model, moves it to the "Production" stage by hand and serves it.
This version rebuilds the MLflow side of the lab: a real tracking server, full run logging
(datasets, plots, tables, nested tuning runs), a model registry workflow that promotes a model
**only if it passes automatic checks against the current production model**, a custom model
that accepts raw input, batch inference with drift tracking, MLflow Projects, and tests.

The data is the [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) dataset:
45,211 phone calls from a Portuguese bank's marketing campaigns (May 2008 – Nov 2010).
The task is to predict whether the client subscribes to a term deposit (`y = yes`, 11.7% of calls).

## Source

The `original/` folder contains the unmodified lab files from the course repository
(`README.md`, `starter.ipynb`, `requirements.txt`).

## My changes

### MLflow changes (the core of the lab)

| | Original lab | This submission |
|---|---|---|
| **Tracking backend** | Default local `./mlruns` folder, no server | `mlflow server` with a **SQLite backend store** and a separate **artifact store** (`mlartifacts/`). All scripts talk to it over HTTP and stop with a clear message if it is not running |
| **What a run logs** | `n_estimators` and `auc` | All hyperparameters, the tuned decision threshold, 8 validation metrics (ROC AUC, PR AUC, F1, precision, recall, accuracy, predicted vs. actual positive rate), a ROC / precision-recall / confusion-matrix figure (`log_figure`), a feature-importance table (`log_table`), and tags (`family`, `kind`, `candidate`) |
| **Dataset tracking** | None | Train / validation / test / new-batch data logged with `mlflow.data.from_pandas` + `mlflow.log_input`, so every run shows exactly which data it used and in which role (training, validation, test, inference) |
| **Hyperparameter tuning** | Hyperopt with `SparkTrials` (needs Spark + Java) | Hyperopt TPE with plain `Trials`. One **parent run** per search with one **nested child run per trial**, plus a `best_val_roc_auc_so_far` metric logged per step to show convergence. Three model families with their own search spaces: logistic regression, random forest, histogram gradient boosting |
| **Logged model** | Wrapper that only returns `predict_proba` | Custom `mlflow.pyfunc.PythonModel` (`BankMarketingModel`) that bundles the preprocessing pipeline **and** the F1-tuned decision threshold. It accepts raw customer rows (including missing values) and returns `probability` and `prediction`. Logged with a signature, an input example, pinned pip requirements, and `code_paths` so it loads on any machine |
| **Model registry** | `transition_model_version_stage(... "Production")` (stages are deprecated in MLflow 3) | **Aliases**: `@challenger` for the version under test, `@champion` for production. Models are loaded as `models:/bank-marketing-classifier@champion` |
| **Promotion** | Done by hand, no checks | `promote.py` registers the best candidate run, scores challenger **and** current champion on the held-out test set, and applies **gates**: ROC AUC ≥ 0.68, recall ≥ 0.25, and ROC AUC at least +0.002 over the champion. Only a passing model gets `@champion`. Each attempt is its own MLflow run with both models' metrics, a side-by-side ROC plot and `gate_results.json`; versions are tagged `validation_status=passed/failed` |
| **Batch inference** | `spark_udf` on a Delta table (not runnable locally) | `batch_predict.py` scores the newest 20% of calls with `@champion`, logs overall metrics, and logs **per-time-slice metrics as steps** so the MLflow UI draws a line chart of performance and positive rate over time (drift). Predictions CSV saved as a run artifact |
| **Real-time serving** | `mlflow models serve` on a run path | Serves the registry alias `models:/bank-marketing-classifier@champion`; `predict_client.py` sends raw rows (with `null`s) and prints the answers. Bad input is rejected by the model signature (HTTP 400) |
| **MLflow Projects** | None | `MLproject` + `python_env.yaml` with `train`, `promote` and `batch_predict` entry points, runnable with `mlflow run` |
| **Testing / CI** | None | 17 pytest tests, including an end-to-end MLflow test (train → reject → promote → nested runs → champion defended → batch inference) against a throwaway SQLite store. GitHub Actions runs them on every push |

### Data and model changes

| | Original lab | This submission |
|---|---|---|
| **Dataset** | Wine Quality (red + white), quality ≥ 7 = "high quality" | UCI Bank Marketing, downloaded automatically on first run |
| **Leakage** | – | `duration` (call length) is dropped: it is only known *after* the call ends, so it would leak the answer |
| **Split** | Random 60 / 20 / 20 | Oldest 80% of calls → stratified 60 / 20 / 20 train / val / test. **Newest 20%** held back as a "new batch" that the model never sees, used for batch inference |
| **Preprocessing** | None needed (all numeric) | `ColumnTransformer`: scaling for 6 numeric columns, one-hot encoding for 9 categorical columns, missing categories filled with `"unknown"` – all inside the logged model |
| **Models** | Random forest, XGBoost | Logistic regression, random forest, scikit-learn `HistGradientBoostingClassifier` (replaces XGBoost, which needs the extra `libomp` system library on macOS) |
| **Decision threshold** | Fixed 0.5 | Chosen on validation to maximise F1 (only ~7% positives, so 0.5 would predict almost no "yes") |

## Project layout

| File | What it does |
|---|---|
| `bankml/data.py` | Download, clean and split the dataset; `prepare_features` (shared by training and the served model) |
| `bankml/model.py` | Model families, Hyperopt search spaces, metrics, plots, feature importance, the `BankMarketingModel` pyfunc |
| `bankml/tracking.py` | Tracking-server connection, experiment / model / alias names, dataset logging |
| `train.py` | Train one model family: a baseline run, or a Hyperopt search with nested runs |
| `promote.py` | Register the best candidate, compare it with the champion, promote only if it passes the gates |
| `batch_predict.py` | Batch inference with `@champion` on the newest calls, with per-slice drift metrics |
| `predict_client.py` | Send raw rows to the REST model server and print the predictions |
| `MLproject`, `python_env.yaml` | MLflow Project entry points |
| `tests/` | Unit tests for data and model code, end-to-end MLflow workflow test |
| `.github/workflows/tests.yml` | Runs the tests on GitHub Actions |
| `original/` | Unmodified course lab files |

## 1. What you need to install

| Tool | Version | Notes |
|---|---|---|
| Python | **3.13** (tested on 3.13.15) | Download from [python.org](https://www.python.org/downloads/). On Windows, tick "Add python.exe to PATH" in the installer |
| Git | any recent version | To clone the repo ([git-scm.com](https://git-scm.com/downloads)) |
| Web browser | any | For the MLflow UI |

No Java, Spark, Docker, conda or GPU needed. All Python packages are pinned in `requirements.txt`
(MLflow 3.17.2, scikit-learn 1.9.1, hyperopt 0.3.0, pandas 3.1.0, numpy 2.5.4, matplotlib 3.11.2,
requests 2.34.2, pytest 9.1.1). An internet connection is needed once, to install packages and
download the dataset (≈ 3 MB).

Check Python:

```bash
python3 --version      # macOS / Linux  -> Python 3.13.x
python --version       # Windows
```

## 2. Set up the environment

```bash
git clone https://github.com/Vishwapujara/mlops-lab2-mlflow-bank-marketing.git
cd mlops-lab2-mlflow-bank-marketing

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Every command below assumes the virtual environment is **active** (your prompt starts with `(.venv)`).
Open new terminals in the project folder and activate it there too.

Optional check that everything installed correctly (takes ~10 s, works offline):

```bash
pytest
# ... 17 passed
```

## 3. Start the MLflow tracking server (terminal 1)

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts \
              --host 127.0.0.1 --port 5050
```

Leave it running. Open **http://127.0.0.1:5050** in a browser to see the MLflow UI.
Runs and registered models are stored in `mlflow.db`; models, plots and files in `mlartifacts/`.

> Port 5050 is used because macOS reserves port 5000 for AirPlay Receiver. To use a different
> port, change `--port` and set `export MLFLOW_TRACKING_URI=http://127.0.0.1:<port>` in the other terminals.

## 4. Run the workflow (terminal 2)

Activate the environment in a second terminal, then run these steps in order.
The numbers below are from a real run; they are reproducible because all seeds are fixed.

### 4a. Baseline – and the gates reject it

```bash
python train.py --family rf      # untuned random forest, 10 trees (same as the original lab)
python promote.py
```

The first run downloads the dataset into `data/`. Expected output:

```
Run 0d11dae1... (rf-baseline)
  params:    {'n_estimators': 10, 'max_depth': None, 'min_samples_leaf': 1, 'class_weight': None}
  threshold: 0.300
  val_roc_auc   0.6370
  ...
Candidate run 0d11dae1... -> bank-marketing-classifier v1 (newly registered)
{
  "min_roc_auc": {"passed": false, "value": 0.6517, "required": 0.68},
  "min_recall":  {"passed": true,  "value": 0.2819, "required": 0.25}
}
REJECTED: v1 stays @challenger; no champion yet
```

The baseline is registered as version 1 but **does not** become the champion: its test ROC AUC
is below 0.68.

### 4b. Tune gradient boosting – it becomes the champion

```bash
python train.py --family hgb --max-evals 25     # ~15 s
python promote.py
```

```
Run 2cab89fe... (hgb-tuned)
  params:    {'l2_regularization': 3.41, 'learning_rate': 0.117, 'max_iter': 300, 'max_leaf_nodes': 16, 'min_samples_leaf': 80}
  threshold: 0.142
  val_roc_auc   0.7049
  ...
Candidate run 2cab89fe... -> bank-marketing-classifier v2 (newly registered)
{
  "min_roc_auc": {"passed": true, "value": 0.7140, "required": 0.68},
  "min_recall":  {"passed": true, "value": 0.3189, "required": 0.25}
}
PROMOTED: v2 is now @champion
```

### 4c. Tune a random forest – the champion defends its title

```bash
python train.py --family rf --max-evals 15      # ~10 s
python promote.py
```

```
Run b5cb52e9... (rf-tuned)
  val_roc_auc   0.7094          <- best validation score so far, so promote.py picks it
  ...
Candidate run b5cb52e9... -> bank-marketing-classifier v3 (newly registered)
{
  "min_roc_auc":    {"passed": true,  "value": 0.7124,  "required": 0.68},
  "min_recall":     {"passed": true,  "value": 0.2942,  "required": 0.25},
  "beats_champion": {"passed": false, "value": -0.0017, "required": 0.002}
}
REJECTED: v3 stays @challenger; @champion is still v2
```

v3 had the best **validation** score, but on the held-out **test** set it is slightly worse than v2,
so production keeps v2. This is the point of gating on data the candidate was not selected on.

You can also try `python train.py --family logreg --max-evals 15` and promote again; it does not beat the champion either.

Optional flags: `python train.py --help`, `python promote.py --help`
(e.g. `--run-id` to promote a specific run, `--min-roc-auc` / `--min-recall` / `--min-improvement` to change the gates).

### 4d. Batch inference on the newest calls

```bash
python batch_predict.py
```

```
Scored 9043 new calls with bank-marketing-classifier v2 (@champion)
  ROC AUC 0.7019 | PR AUC 0.5292 | recall 0.4478
  actual 'yes' rate 0.316 vs 0.067 in training -> the population has shifted
  predictions saved to .../outputs/new_batch_predictions.csv
```

In the MLflow UI, open the `batch-inference-v2` run and look at its metric charts: `chunk_actual_positive_rate`
climbs from 0.08 to 0.53 across the four time slices, while `chunk_roc_auc` stays around 0.59–0.64.
The later customers behave very differently from the training data – a signal to retrain.

## 5. Serve the champion as a REST API (terminal 3)

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050        # Windows: set MLFLOW_TRACKING_URI=http://127.0.0.1:5050
mlflow models serve -m "models:/bank-marketing-classifier@champion" -h 127.0.0.1 -p 5001 --env-manager local
```

Wait for `Application startup complete.`. Then, back in terminal 2:

```bash
python predict_client.py
```

```
   age            job month poutcome actual  probability prediction
0   53    blue-collar   jul  missing     no        0.032         no
1   47     management   aug  missing     no        0.064         no
2   56  self-employed   aug  missing     no        0.032         no
3   33    blue-collar   nov  missing     no        0.124         no
4   35    blue-collar   apr    other     no        0.052         no
5   35     technician   oct  missing      ?        0.144        yes
```

Rows 0–4 are test-set customers; row 5 is a hand-written customer with a missing `poutcome`.
The model fills missing values itself and returns a probability plus a yes/no decision at its tuned threshold.

The model signature rejects malformed requests:

```bash
curl -X POST http://127.0.0.1:5001/invocations -H 'Content-Type: application/json' \
     -d '{"dataframe_records":[{"age":"abc"}]}'
# {"error_code": "BAD_REQUEST", "message": "Invalid input. Data is not compatible with model signature. Failed to convert column age to type 'int64'..."}
```

Stop the servers with `Ctrl+C` in terminals 1 and 3 when done.

## 6. Running with MLflow Projects (alternative)

The same steps can be launched through the `MLproject` file. With the tracking server running:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
mlflow run . -e train -P family=hgb -P max_evals=25 --env-manager local --experiment-name bank-marketing
mlflow run . -e promote --env-manager local --experiment-name bank-marketing
mlflow run . -e batch_predict --env-manager local --experiment-name bank-marketing
```

Each ends with `=== Run (ID '...') succeeded ===`. Drop `--env-manager local` to let MLflow build
a fresh virtual environment from `python_env.yaml` (slower, needs `pyenv`).

## 7. What a successful run looks like in the MLflow UI

At **http://127.0.0.1:5050**:

- **Experiments → bank-marketing**: runs `rf-baseline`, `hgb-tuned`, `rf-tuned`, `promote-v1/v2/v3`, `batch-inference-v2`.
  Expanding `hgb-tuned` shows its 25 nested `hgb-trial` runs. Select the trials → **Compare** for a parallel-coordinates plot of the search.
- **A training run**: Parameters, Metrics (`val_*`), Datasets (`bank-train`, `bank-val`), and Artifacts:
  `model/`, `plots/validation.png` (ROC, PR, confusion matrix), `feature_importance.json`.
- **A promotion run**: `challenger_test_*` and `champion_test_*` metrics, `plots/challenger_vs_champion.png`, `gate_results.json`.
- **Models → bank-marketing-classifier**: versions 1–3, alias `@champion` on v2 and `@challenger` on v3,
  tags `validation_status` and `test_roc_auc` on each version.

## Why the scores are around 0.70 ROC AUC

Published results for this dataset reach ~0.80 ROC AUC without `duration` on a random split of all rows
(this code gets 0.80 that way too). Here the models train only on the **oldest 80%** of calls, where just
10% of clients have a known previous-campaign outcome (`poutcome`, the strongest signal) and only 6.7%
subscribe. In the newest 20% those numbers are 49% and 31.6%. Training on the past and checking on the
future is harder, but it is the realistic setting – and it is what makes the drift tracking in step 4d meaningful.

## Running the tests

```bash
pytest -v
```

`tests/test_mlflow_workflow.py` runs the whole MLflow workflow on synthetic data against a temporary SQLite
tracking store, so it needs neither the tracking server nor the internet. GitHub Actions runs the same tests on every push.

## Starting over

Stop the servers, then delete the local state:

```bash
rm -rf mlflow.db mlartifacts outputs
```
