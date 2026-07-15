"""Training pipeline for the multi-disorder eating disorder classifier.

Trains two class-balanced, interpretable models -- logistic regression and a
random forest -- on the engineered feature set from features.py. Splits on
the RAW patient data (before feature engineering) and saves the raw test
rows, so evaluate.py can featurize both this held-out set and a separately
generated realistic-prevalence set through the exact same code path.
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features import LABEL_COL, build_preprocessor, prepare_dataframe


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                ("classify", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "classify",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=10,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the eating disorder classifier models.")
    parser.add_argument("--data", type=str, default="data/synthetic_patients.csv")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_df = pd.read_csv(args.data)
    train_raw, test_raw = train_test_split(
        raw_df, test_size=args.test_size, random_state=args.seed, stratify=raw_df[LABEL_COL]
    )
    X_train, y_train = prepare_dataframe(train_raw)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    models = build_models()
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, model_dir / f"{name}.joblib")
        print(f"Trained and saved {name} -> {model_dir / f'{name}.joblib'}")

    test_path = Path(args.data).parent / "test_set.csv"
    test_raw.to_csv(test_path, index=False)
    print(f"Saved held-out raw test set ({len(test_raw)} rows) -> {test_path}")


if __name__ == "__main__":
    main()
