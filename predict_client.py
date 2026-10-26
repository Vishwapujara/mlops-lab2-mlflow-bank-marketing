"""Send raw customer rows to the served model and print its answers.

Start the server first (README, step 7):
  mlflow models serve -m "models:/bank-marketing-classifier@champion" -p 5001 --env-manager local

  python predict_client.py            # 5 test-set customers + 1 hand-written one
  python predict_client.py --n 10 --url http://127.0.0.1:5001/invocations
"""

import argparse

import pandas as pd
import requests

from bankml.data import FEATURES, labels, load_splits

# A customer typed in by hand. Missing values are allowed: the model fills them in.
NEW_CUSTOMER = {
    "age": 35, "balance": 1500, "day": 15, "campaign": 1, "pdays": 90, "previous": 2,
    "job": "technician", "marital": "single", "education": "tertiary", "default": "no",
    "housing": "no", "loan": "no", "contact": "cellular", "month": "oct", "poutcome": None,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:5001/invocations")
    parser.add_argument("--n", type=int, default=5, help="test-set rows to send")
    args = parser.parse_args(argv)

    test = load_splits()["test"].sample(args.n, random_state=0)
    rows = pd.concat([test[FEATURES], pd.DataFrame([NEW_CUSTOMER])], ignore_index=True)
    # NaN is not valid JSON, so send missing values as null.
    records = rows.astype(object).where(rows.notna(), None).to_dict(orient="records")

    try:
        response = requests.post(args.url, json={"dataframe_records": records}, timeout=60)
    except requests.ConnectionError:
        raise SystemExit(f"No model server at {args.url}. Start it first (README, step 7).")
    response.raise_for_status()
    predictions = pd.DataFrame(response.json()["predictions"])

    report = rows[["age", "job", "month", "poutcome"]].fillna("missing")
    report["actual"] = list(labels(test).map({1: "yes", 0: "no"})) + ["?"]
    report["probability"] = predictions["probability"].round(3)
    report["prediction"] = predictions["prediction"].map({1: "yes", 0: "no"})
    print(report.to_string())
    return predictions


if __name__ == "__main__":
    main()
