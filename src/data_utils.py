"""
data_utils.py
--------------
Loads the UCI 'student-mat.csv' dataset and engineers it into the five
features used throughout this project:

    attendance_pct    (0-100)   derived from `absences`
    internal_marks    (0-100)   derived from `G1` (first period grade)
    assignment_score  (0-100)   derived from `G2` (second period grade)
    study_hours       (hrs/day) derived from `studytime` (weekly study band)
    previous_cgpa     (0-10)    derived from average(G1, G2)

Target:
    performance_score (0-100)  derived from `G3` (final grade)

NOTE (transparency): the original UCI dataset does not literally contain
"attendance %", "assignment score" or "CGPA" columns. These are reasonable,
documented proxy mappings built from the closest available columns so that
the project can be trained/evaluated on a real, publicly available dataset
rather than fully synthetic numbers. The mapping is documented here and in
the research paper draft.
"""

import numpy as np
import pandas as pd
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "student-mat.csv")

FEATURE_NAMES = [
    "attendance_pct",
    "internal_marks",
    "assignment_score",
    "study_hours",
    "previous_cgpa",
]

STUDY_HOURS_MAP = {1: 1.0, 2: 3.5, 3: 7.5, 4: 12.0}


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    max_abs = max(df["absences"].max(), 1)
    out["attendance_pct"] = (100 - (df["absences"] / max_abs * 100)).clip(0, 100)

    out["internal_marks"] = (df["G1"] / 20 * 100).clip(0, 100)
    out["assignment_score"] = (df["G2"] / 20 * 100).clip(0, 100)
    out["study_hours"] = df["studytime"].map(STUDY_HOURS_MAP).astype(float)
    out["previous_cgpa"] = ((df["G1"] + df["G2"]) / 2 / 20 * 10).clip(0, 10)

    out["performance_score"] = (df["G3"] / 20 * 100).clip(0, 100)
    out["pass_fail"] = (df["G3"] >= 10).astype(int)

    return out


def get_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    return engineer_features(load_raw(path))


# ---- normalisation helpers (0-1) used by the neural network -------------

FEATURE_RANGES = {
    "attendance_pct": (0, 100),
    "internal_marks": (0, 100),
    "assignment_score": (0, 100),
    "study_hours": (0, 12),
    "previous_cgpa": (0, 10),
}
TARGET_RANGE = (0, 100)


def normalize_features(X: np.ndarray) -> np.ndarray:
    Xn = np.zeros_like(X, dtype=float)
    for i, name in enumerate(FEATURE_NAMES):
        lo, hi = FEATURE_RANGES[name]
        Xn[:, i] = (X[:, i] - lo) / (hi - lo)
    return np.clip(Xn, 0, 1)


def normalize_target(y: np.ndarray) -> np.ndarray:
    lo, hi = TARGET_RANGE
    return np.clip((y - lo) / (hi - lo), 0, 1)


def denormalize_target(yn: np.ndarray) -> np.ndarray:
    lo, hi = TARGET_RANGE
    return yn * (hi - lo) + lo


def train_test_split(X, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


if __name__ == "__main__":
    d = get_dataset()
    print(d.describe())
    print(d.head())
