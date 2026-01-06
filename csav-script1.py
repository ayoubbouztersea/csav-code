#!/usr/bin/env python3
"""Train a classifier that predicts how many articles (0, 1, 2, or 3+) are linked to each dossier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_XGBOOST = False


DATA_DEFAULT = Path("data/DF_ML_M1.csv")
MODEL_DEFAULT = Path("models/article_count_classifier.joblib")
METRICS_DEFAULT = Path("models/article_count_metrics.json")

NUMERIC_FEATURES = []
CATEGORICAL_FEATURES = [
    "Do",
    "s Code_Pdt",
    "Fam1",
    "Fam2",
    "Fam3",
    "Fam4",
    "Marque",
    "Symptome",
    "CodeError",
    "Type_Prediag",
    "Version",
]

CLASSIFIER_CHOICES = [
    "sgd",
    "logreg",
    "linear_svc",
    "passive_aggressive",
    "xgboost"

]
ARTICLE_COUNT_LABELS = [0, 1, 2, 3]
MAX_ARTICLE_COUNT = ARTICLE_COUNT_LABELS[-1]
TARGET_COLUMN = "len_LIBEL_ARTICLE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DEFAULT,
        help="Path to the input CSV file (default: %(default)s).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_DEFAULT,
        help="Path where the trained pipeline will be saved (default: %(default)s).",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=METRICS_DEFAULT,
        help="Optional path to store evaluation metrics as JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for evaluation (default: %(default)s).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum training iterations for iterative linear classifiers (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="Regularization strength for SGDClassifier (default: %(default)s).",
    )
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIER_CHOICES + ["all"],
        default="sgd",
        help="Which classifier to train. Use 'all' to train and compare every available option.",
    )
    return parser.parse_args()


def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find data file at {path}")

    df = pd.read_csv(
        path,
        dtype={"Dossier": "string"},
        low_memory=False,
    )
    df["Dossier"] = df["Dossier"].str.strip()
    if "CODE_ARTICLE" in df.columns:
        df["CODE_ARTICLE"] = df["CODE_ARTICLE"].fillna("NO_PART").astype("string").str.strip()
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    return df


def build_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_frame = (
        df[["Dossier", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]]
        .drop_duplicates(subset="Dossier", keep="first")
        .set_index("Dossier")
    )

    if TARGET_COLUMN in df.columns:
        target_series = pd.to_numeric(
            df[["Dossier", TARGET_COLUMN]]
            .drop_duplicates(subset="Dossier", keep="first")
            .set_index("Dossier")[TARGET_COLUMN],
            errors="coerce",
        ).fillna(0).astype(int)
    elif "CODE_ARTICLE" in df.columns:
        article_counts = (
            df.loc[df["CODE_ARTICLE"] != "NO_PART"]
            .groupby("Dossier")["CODE_ARTICLE"]
            .nunique()
        )
        target_series = article_counts.reindex(feature_frame.index).fillna(0).astype(int)
    else:
        raise ValueError(
            "Could not determine target labels: expected either "
            f"'{TARGET_COLUMN}' or 'CODE_ARTICLE' columns in the dataset."
        )

    target = target_series.clip(lower=0, upper=MAX_ARTICLE_COUNT).astype(int)
    return feature_frame, target


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def _ensure_dense(X):
    """Convert sparse matrices to dense arrays when required by downstream estimators."""
    return X.toarray() if hasattr(X, "toarray") else X


def build_model(estimator, requires_dense: bool = False) -> Pipeline:
    preprocessor = build_preprocessor()

    steps = [("preprocessor", preprocessor)]
    if requires_dense:
        steps.append(("to_dense", FunctionTransformer(_ensure_dense, accept_sparse=True)))
    steps.append(("classifier", estimator))

    return Pipeline(steps=steps)


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred, labels=ARTICLE_COUNT_LABELS)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
        "labels": ARTICLE_COUNT_LABELS,
    }
    return metrics


def make_classifier(name: str, args: argparse.Namespace) -> Tuple[object, bool]:
    if name == "sgd":
        return SGDClassifier(
            loss="log_loss",
            alpha=args.alpha,
            penalty="l2",
            max_iter=args.max_iter,
            early_stopping=True,
            n_iter_no_change=5,
            class_weight="balanced",
            random_state=args.random_state,
        ), False
    if name == "logreg":
        return LogisticRegression(
            solver="saga",
            penalty="l2",
            class_weight="balanced",
            max_iter=args.max_iter,
            n_jobs=-1,
            multi_class="multinomial",
            random_state=args.random_state,
        ), False
    if name == "ridge":
        return RidgeClassifier(class_weight="balanced"), False
    if name == "linear_svc":
        return LinearSVC(
            class_weight="balanced",
            max_iter=args.max_iter,
            random_state=args.random_state,
            dual=False,
        ), False
    if name == "passive_aggressive":
        return PassiveAggressiveClassifier(
            class_weight="balanced",
            max_iter=args.max_iter,
            random_state=args.random_state,
        ), False
    if name == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError(
                "xgboost is not installed. Please install it with `pip install xgboost` to use this classifier."
            )
        return (
            XGBClassifier(
                objective="multi:softprob",
                num_class=len(ARTICLE_COUNT_LABELS),
                learning_rate=0.05,
                n_estimators=500,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=args.random_state,
                n_jobs=-1,
            ),
            False,
        )
    if name == "random_forest":
        return (
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                class_weight="balanced_subsample",
                random_state=args.random_state,
                n_jobs=-1,
            ),
            True,
        )
    if name == "gradient_boosting":
        return (
            GradientBoostingClassifier(
                learning_rate=0.05,
                n_estimators=400,
                max_depth=3,
                random_state=args.random_state,
            ),
            True,
        )
    if name == "extra_trees":
        return (
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=None,
                class_weight="balanced",
                random_state=args.random_state,
                n_jobs=-1,
            ),
            True,
        )
    raise ValueError(f"Unknown classifier requested: {name}")


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.data_path} ...")
    raw_df = load_raw_data(args.data_path)

    X, y = build_dataset(raw_df)
    print(
        f"Target distribution (after clipping >= {MAX_ARTICLE_COUNT} to {MAX_ARTICLE_COUNT}):",
        y.value_counts().sort_index().to_dict(),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # Export test dataset as CSV
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test
    test_df_path = Path("test_df_count_model.csv")
    test_df.to_csv(test_df_path)
    print(f"\nExported test dataset to {test_df_path} ({len(test_df)} rows)")

    classifier_names = CLASSIFIER_CHOICES if args.classifier == "all" else [args.classifier]
    metrics_by_classifier: Dict[str, dict] = {}
    model_paths: Dict[str, str] = {}
    best_name = None
    best_score = float("-inf")

    for name in classifier_names:
        print(f"\nFitting classifier '{name}' ...")
        estimator, requires_dense = make_classifier(name, args)
        pipeline = build_model(estimator, requires_dense=requires_dense)
        pipeline.fit(X_train, y_train)

        print(f"Evaluating '{name}' on hold-out set ...")
        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics["classifier"] = name
        metrics_by_classifier[name] = metrics
        print(json.dumps({k: v for k, v in metrics.items() if k not in {"classification_report", "confusion_matrix"}}, indent=2))
        print("Classification report:")
        print(classification_report(y_test, pipeline.predict(X_test), digits=3))
        print("Confusion matrix (rows=true, cols=pred):")
        print(pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"]))

        if len(classifier_names) == 1:
            model_path = args.model_path
        else:
            model_path = args.model_path.with_name(f"{args.model_path.stem}_{name}{args.model_path.suffix}")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        model_paths[name] = str(model_path)
        print(f"Saved trained pipeline for '{name}' to {model_path}")

        score = metrics["balanced_accuracy"]
        if score > best_score:
            best_score = score
            best_name = name

    if args.metrics_path:
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_path.open("w", encoding="utf-8") as fh:
            if len(classifier_names) == 1:
                json.dump(metrics_by_classifier[classifier_names[0]], fh, indent=2)
            else:
                payload = {
                    "best_classifier": best_name,
                    "best_balanced_accuracy": best_score,
                    "metrics": metrics_by_classifier,
                    "model_paths": model_paths,
                }
                json.dump(payload, fh, indent=2)
        print(f"\nStored metrics at {args.metrics_path}")

    if len(classifier_names) > 1:
        print(f"\nBest classifier based on balanced accuracy: '{best_name}' ({best_score:.4f})")


if __name__ == "__main__":
    main()
