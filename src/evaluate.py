"""Evaluation for the multi-disorder eating disorder classifier.

Reports standard classification metrics plus feature importances, and
specifically highlights recall on the atypical_anorexia class: Sick
Enough's central clinical argument is that atypical AN patients are
routinely dismissed as "not sick enough" because they aren't underweight
(Ch. 1, p.842-861), so a model that quietly trades away recall on that
class for overall accuracy would reproduce exactly the bias the book is
about.

Two evaluation modes are supported:
  1. Balanced (default): the held-out split of the training-distribution
     data, saved by model.py to data/test_set.csv.
  2. Realistic-prevalence (--prevalence-eval-data): a separately generated
     set sampled at approximate true population prevalence
     (`python -m src.data_gen --prevalence`), run through the SAME already-
     trained models. Precision on rare classes is expected to drop
     substantially here -- that's the point, not a bug: it's an honest
     picture of deployment-time performance under real class imbalance,
     which balanced-eval numbers alone would hide.
"""

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

from src.features import LABELS, prepare_dataframe


def _plot_confusion_matrix(y_test, y_pred, model_name: str, suffix: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=LABELS, ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"Confusion matrix: {model_name} ({suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{model_name}_{suffix}.png", dpi=150)
    plt.close(fig)


def _plot_feature_importance(pipeline, model_name: str, out_dir: Path, top_n: int = 20) -> None:
    classifier = pipeline.named_steps["classify"]
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        title = f"Top {top_n} feature importances: {model_name}"
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_).mean(axis=0)
        title = f"Top {top_n} |mean coefficient|: {model_name}"
    else:
        return

    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(x=importances[order], y=[feature_names[i] for i in order], ax=ax, color="steelblue")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / f"feature_importance_{model_name}.png", dpi=150)
    plt.close(fig)


def evaluate_model(
    pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str, suffix: str, out_dir: Path
) -> None:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    present_labels = [label for label in LABELS if (y_test == label).any()]

    print(f"\n{'=' * 60}\n{model_name} -- {suffix}\n{'=' * 60}")
    report = classification_report(y_test, y_pred, labels=present_labels, digits=3, zero_division=0)
    print(report)

    # predict_proba columns follow the fitted classifier's classes_ order
    # (alphabetical), which is not necessarily LABELS -- must match for AUC.
    proba_classes = list(pipeline.named_steps["classify"].classes_)
    try:
        y_bin = label_binarize(y_test, classes=proba_classes)
        auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
        print(f"Macro-average ROC-AUC (one-vs-rest): {auc:.3f}")
    except ValueError as exc:
        print(f"ROC-AUC skipped: {exc}")

    if "atypical_anorexia" in present_labels:
        report_dict = classification_report(y_test, y_pred, labels=present_labels, output_dict=True, zero_division=0)
        aan_recall = report_dict["atypical_anorexia"]["recall"]
        print(
            f"atypical_anorexia recall: {aan_recall:.3f} "
            "(fraction of true AAN patients the model correctly flags -- the clinically critical number)"
        )

    _plot_confusion_matrix(y_test, y_pred, model_name, suffix, out_dir)
    if suffix == "balanced":
        _plot_feature_importance(pipeline, model_name, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained eating disorder classifiers.")
    parser.add_argument("--test-data", type=str, default="data/test_set.csv")
    parser.add_argument("--prevalence-eval-data", type=str, default=None, help="optional realistic-prevalence CSV (see data_gen.py --prevalence)")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--report-dir", type=str, default="reports")
    args = parser.parse_args()

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test, y_test = prepare_dataframe(pd.read_csv(args.test_data))

    model_dir = Path(args.model_dir)
    for model_path in sorted(model_dir.glob("*.joblib")):
        pipeline = joblib.load(model_path)
        evaluate_model(pipeline, X_test, y_test, model_path.stem, "balanced", out_dir)

    if args.prevalence_eval_data:
        X_prev, y_prev = prepare_dataframe(pd.read_csv(args.prevalence_eval_data))
        print("\n" + "#" * 60)
        print("# REALISTIC-PREVALENCE EVALUATION (not the balanced numbers above)")
        print("#" * 60)
        for model_path in sorted(model_dir.glob("*.joblib")):
            pipeline = joblib.load(model_path)
            evaluate_model(pipeline, X_prev, y_prev, model_path.stem, "prevalence", out_dir)

    print(f"\nSaved confusion matrix and feature importance plots -> {out_dir}/")


if __name__ == "__main__":
    main()
