"""Load, clean and split the UCI Bank Marketing dataset.

The raw file has one row per phone call, in date order (May 2008 - Nov 2010).
The first 80% of calls is the "historical" data, split randomly into train,
validation and test. The last 20% is held back as a "new batch" that arrives
after the model is deployed, used for batch inference and drift checks.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://archive.ics.uci.edu/static/public/222/data.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "bank-full.csv"

TARGET = "y"
# Call length is only known after the call ends, so using it leaks the answer.
LEAKY_COLUMNS = ["duration"]
NUMERIC = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "poutcome"]
FEATURES = NUMERIC + CATEGORICAL

NEW_BATCH_FRACTION = 0.2
SEED = 42


def load_dataset(path=RAW_PATH):
    """Return the full dataset, downloading it on first use."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Bank Marketing data from {DATA_URL}")
        pd.read_csv(DATA_URL).to_csv(path, index=False)
    df = pd.read_csv(path)
    # UCI calls this column day_of_week, but it holds the day of the month (1-31).
    return df.rename(columns={"day_of_week": "day"})


def prepare_features(df):
    """Select the model's input columns and fill missing categories.

    Used both in training and inside the logged MLflow model, so served
    requests get exactly the same cleaning as the training data.
    """
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    X = df[FEATURES].copy()
    X[CATEGORICAL] = X[CATEGORICAL].astype(object).where(X[CATEGORICAL].notna(), "unknown")
    X[CATEGORICAL] = X[CATEGORICAL].astype(str)
    return X


def labels(df):
    """Target as 0/1 (1 = client subscribed to a term deposit)."""
    return (df[TARGET] == "yes").astype(int)


def split(df, seed=SEED):
    """Split into train / val / test (historical) and new_batch (most recent calls)."""
    n_hist = int(len(df) * (1 - NEW_BATCH_FRACTION))
    hist, new_batch = df.iloc[:n_hist], df.iloc[n_hist:]
    train, rest = train_test_split(hist, train_size=0.6, stratify=labels(hist), random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=labels(rest), random_state=seed)
    return {"train": train, "val": val, "test": test, "new_batch": new_batch}


def load_splits():
    return split(load_dataset())
