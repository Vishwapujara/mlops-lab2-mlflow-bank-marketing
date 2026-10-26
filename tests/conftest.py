import numpy as np
import pandas as pd
import pytest


def make_bank_df(n=2000, seed=0):
    """Synthetic data with the real dataset's columns, so tests run offline and fast."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 90, n),
        "job": rng.choice(["admin.", "technician", "services", "management", None], n),
        "marital": rng.choice(["married", "single", "divorced"], n),
        "education": rng.choice(["primary", "secondary", "tertiary", None], n),
        "default": rng.choice(["no", "yes"], n, p=[0.95, 0.05]),
        "balance": rng.integers(-500, 5000, n),
        "housing": rng.choice(["no", "yes"], n),
        "loan": rng.choice(["no", "yes"], n),
        "contact": rng.choice(["cellular", "telephone", None], n),
        "day": rng.integers(1, 32, n),
        "month": rng.choice(["may", "jun", "jul", "aug"], n),
        "duration": rng.integers(10, 600, n),
        "campaign": rng.integers(1, 10, n),
        "pdays": rng.choice([-1, 30, 90, 180], n),
        "previous": rng.integers(0, 5, n),
        "poutcome": rng.choice(["success", "failure", "other", None], n, p=[0.1, 0.1, 0.05, 0.75]),
    })
    signal = 2.5 * (df.poutcome == "success") + (df.age > 60) + 0.3 * (df.housing == "no")
    df["y"] = np.where(signal + rng.normal(0, 1, n) > 1.8, "yes", "no")
    return df


@pytest.fixture
def bank_df():
    return make_bank_df()
