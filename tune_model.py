"""
Tune WEIGHTED XGBoost for the Steam price predictor.

Uses the same 70/15/15 split (random_state=42) as train_model.py.
Only the 70% TRAIN partition is used in 3-fold cross-validation. The 15%
VALIDATION partition is used once, after choosing hyperparameters via CV.
The final 15% TEST partition is never evaluated or fitted on.

Fit the multi-label encoders and scaler *inside each cross-validation fold*.
This avoids learning validation-fold vocabularies during preprocessing.

Run from the project folder containing test.csv:
    pip install pandas numpy scipy scikit-learn xgboost joblib
    python tune_model.py

Optional:
    python tune_model.py --trials 48 --folds 3
    python tune_model.py --trials 8 --folds 3   # shorter trial run
    python tune_model.py --dataset path/to/test.csv --outdir tuning_results

Output (saved after each completed configuration):
    tuning_results/tuning_trials.csv
    tuning_results/best_cv_configuration.json
    tuning_results/tuning_cv_folds.csv
    tuning_results/validation_summary.csv       (after the search completes)
    tuning_results/validation_by_price_range.csv (after the search completes)
    tuning_results/validation_predictions.csv    (after the search completes)
    tuning_results/tuned_xgboost_candidate.joblib (separate from the website model)

Main selection score (LOWER IS BETTER):
    70% * cross-validated MAE for games priced $20+
    30% * cross-validated MAE for all games

This emphasizes premium games without completely ignoring ordinary prices.
Weights are applied ONLY to training rows, based on their known target prices.
No dataset rows are deleted in this script. Suspicious source prices remain
an important limitation of the available labels.
"""

import argparse
import json
import math
import os
import random
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from xgboost import XGBRegressor

SEED = 42
LIST_COLUMNS = [
    "popular_tags", "genre", "game_details", "publisher", "developer", "languages"
]
NUMERIC_COLUMNS = ["achievements", "release_year"]
BINS = [0, 10, 20, 40, float("inf")]
BAND_LABELS = ["$0-10", "$10-20", "$20-40", "$40+"]

# Company names in validation folds are often unseen during training. That is
# normal and would otherwise produce an enormous warning for every fold.
warnings.filterwarnings(
    "ignore", message=r"unknown class\(es\).*will be ignored", category=UserWarning
)


def parse_labels(series):
    """Exactly matches train_model.py: comma-separated, stripped labels."""
    return series.fillna("").astype(str).apply(
        lambda item: [p.strip() for p in item.split(",") if p.strip()]
    )


class FeaturePreprocessor:
    """Multilabel + numeric preprocessing fitted on training rows only."""

    def __init__(self, list_columns):
        self.list_columns = list(list_columns)
        self.encoders = {}
        self.scaler = StandardScaler()

    def fit_transform(self, rows):
        parts = []
        for column in self.list_columns:
            encoder = MultiLabelBinarizer(sparse_output=True)
            arr = encoder.fit_transform(parse_labels(rows[column]))
            self.encoders[column] = encoder
            parts.append(arr)
        numeric = rows[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="raise")
        if numeric.isna().any().any():
            raise ValueError("Missing achievement/release_year values in training CSV.")
        parts.append(csr_matrix(self.scaler.fit_transform(numeric)))
        return hstack(parts, format="csr", dtype=np.float32)

    def transform(self, rows):
        parts = [
            self.encoders[column].transform(parse_labels(rows[column]))
            for column in self.list_columns
        ]
        numeric = rows[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="raise")
        if numeric.isna().any().any():
            raise ValueError("Missing achievement/release_year values in validation CSV.")
        parts.append(csr_matrix(self.scaler.transform(numeric)))
        return hstack(parts, format="csr", dtype=np.float32)


def sample_weights(prices, weight20, weight40):
    """Training-only weighting; $40+ supersedes the $20+ tier."""
    y = np.asarray(prices, dtype=np.float32)
    return np.where(y >= 40, weight40, np.where(y >= 20, weight20, 1.0))


def model_from_config(config, threads):
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=SEED,
        n_jobs=threads,
        verbosity=0,
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        min_child_weight=float(config["min_child_weight"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        reg_lambda=float(config["reg_lambda"]),
        reg_alpha=float(config["reg_alpha"]),
        gamma=float(config["gamma"]),
    )


def metrics(y, prediction):
    y = np.asarray(y, dtype=float)
    p = np.asarray(prediction, dtype=float).reshape(-1)
    if y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError("Bad prediction vector from XGBoost")
    error = p - y
    absolute = np.abs(error)

    def mean_or_nan(values):
        return float(np.mean(values)) if len(values) else float("nan")

    premium = y >= 20
    mid = (y >= 20) & (y < 40)
    upper = y >= 40
    focus = (y >= 20) & (y <= 60)
    return {
        "overall_mae": float(mean_absolute_error(y, p)),
        "mae_20_plus": mean_or_nan(absolute[premium]),
        "mae_20_to_60": mean_or_nan(absolute[focus]),
        "mae_20_to_40": mean_or_nan(absolute[mid]),
        "mae_40_plus": mean_or_nan(absolute[upper]),
        "bias_40_plus": mean_or_nan(error[upper]),
        "within_5_20_to_60_pct": mean_or_nan(100 * (absolute[focus] <= 5)),
        "n_20_plus": int(premium.sum()),
        "n_40_plus": int(upper.sum()),
    }


def selection_score(all_mae, premium_mae):
    """Lower is better. Premium MAE is the primary objective."""
    return 0.30 * all_mae + 0.70 * premium_mae


def config_key(c):
    return tuple(c[k] for k in [
        "n_estimators", "max_depth", "learning_rate", "min_child_weight",
        "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "gamma",
        "weight20", "weight40"
    ])


def generate_configs(n_trials, seed):
    """A few targeted experiments followed by reproducible random search."""
    base = {
        "n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
        "min_child_weight": 1, "subsample": 0.9,
        "colsample_bytree": 0.8, "reg_lambda": 1.0,
        "reg_alpha": 0.0, "gamma": 0.0,
        "weight20": 2.0, "weight40": 4.0,
    }
    preset = [
        base,
        {**base, "weight20": 1.0, "weight40": 1.0},
        {**base, "weight20": 2.0, "weight40": 8.0},
        {**base, "weight20": 3.0, "weight40": 9.0},
        {**base, "max_depth": 3, "weight20": 2.0, "weight40": 6.0},
        {**base, "max_depth": 5, "weight20": 3.0, "weight40": 9.0},
    ]
    rng = random.Random(seed)
    configs = []
    used = set()
    for c in preset[:n_trials]:
        if config_key(c) not in used:
            used.add(config_key(c))
            configs.append(c)

    tiers = [
        (1.0, 1.0), (1.5, 3.0), (1.5, 6.0), (2.0, 4.0),
        (2.0, 6.0), (2.0, 8.0), (3.0, 6.0), (3.0, 9.0),
        (4.0, 8.0), (4.0, 12.0), (2.0, 12.0)
    ]
    while len(configs) < n_trials:
        w20, w40 = rng.choice(tiers)
        config = {
            "n_estimators": rng.choice([200, 300, 400, 550, 700]),
            "max_depth": rng.choice([2, 3, 4, 5, 6]),
            "learning_rate": rng.choice([0.025, 0.04, 0.06, 0.08, 0.12]),
            "min_child_weight": rng.choice([1, 2, 4, 8]),
            "subsample": rng.choice([0.7, 0.85, 1.0]),
            "colsample_bytree": rng.choice([0.65, 0.8, 1.0]),
            "reg_lambda": rng.choice([1.0, 3.0, 8.0, 15.0]),
            "reg_alpha": rng.choice([0.0, 0.1, 1.0]),
            "gamma": rng.choice([0.0, 0.5, 2.0]),
            "weight20": w20,
            "weight40": w40,
        }
        key = config_key(config)
        if key not in used:
            configs.append(config)
            used.add(key)
    return configs


def save_results(rows, fold_rows, outdir):
    pd.DataFrame(rows).to_csv(outdir / "tuning_trials.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(outdir / "tuning_cv_folds.csv", index=False)
    completed = [r for r in rows if r.get("status") == "ok"]
    if completed:
        best = min(completed, key=lambda r: r["cv_score"])
        content = {
            "selection_rule": "0.70 * CV MAE($20+) + 0.30 * CV MAE(overall)",
            "cv_score": best["cv_score"],
            "cv_overall_mae": best["cv_overall_mae"],
            "cv_mae_20_plus": best["cv_mae_20_plus"],
            "cv_mae_40_plus": best["cv_mae_40_plus"],
            "config": {k: best[k] for k in [
                "n_estimators", "max_depth", "learning_rate", "min_child_weight",
                "subsample", "colsample_bytree", "reg_lambda", "reg_alpha",
                "gamma", "weight20", "weight40"
            ]},
        }
        with open(outdir / "best_cv_configuration.json", "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Cross-validated weighted XGBoost tuning")
    parser.add_argument("--dataset", default="test.csv")
    parser.add_argument("--trials", type=int, default=36, help="Number of configurations (default 36)")
    parser.add_argument("--folds", type=int, default=3, help="CV folds within training data (default 3)")
    parser.add_argument("--threads", type=int, default=4, help="XGBoost CPU threads per trial")
    parser.add_argument("--outdir", default="tuning_results", help="Output directory")
    args = parser.parse_args()
    if args.trials < 1 or args.folds < 2 or args.threads < 1:
        parser.error("--trials and --threads must be >=1, --folds must be >=2")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.dataset}...", flush=True)
    df = pd.read_csv(args.dataset)
    expected = set(LIST_COLUMNS[:-1] + NUMERIC_COLUMNS + ["price"])
    missing = sorted(expected.difference(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    list_columns = [x for x in LIST_COLUMNS if x in df.columns]
    if "languages" not in df.columns:
        print("WARNING: 'languages' absent, skipping that feature. Use current test.csv for language experiment.", flush=True)

    # EXACT same 70/15/15 split as existing train_model.py.
    train, remainder = train_test_split(df, test_size=0.30, random_state=SEED)
    validation, _final_test = train_test_split(
        remainder, test_size=0.50, random_state=SEED
    )
    # The test data is deliberately never accessed again.
    print(
        f"Training {len(train)} | Validation {len(validation)} | "
        f"Untouched TEST {len(_final_test)}", flush=True
    )
    del remainder, _final_test, df

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    y_train = train["price"].to_numpy(dtype=float)
    y_val = validation["price"].to_numpy(dtype=float)
    if not np.isfinite(y_train).all() or not np.isfinite(y_val).all():
        raise ValueError("Dataset contains missing/nonfinite price targets")
    if sum(y_train >= 20) < args.folds:
        raise ValueError("Too few premium games for cross-validation")

    # Cache fold-specific matrices. Every encoder/scaler is fitted ONLY to that
    # fold's training rows. No vocabulary leakage from each held-out CV fold.
    print("Preparing fold-specific preprocessing (no leakage)...", flush=True)
    splits = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    cache = []
    for fold_id, (train_idx, val_idx) in enumerate(splits.split(train), 1):
        fold_train = train.iloc[train_idx]
        fold_val = train.iloc[val_idx]
        preprocessor = FeaturePreprocessor(list_columns)
        X_t = preprocessor.fit_transform(fold_train)
        X_v = preprocessor.transform(fold_val)
        y_t = y_train[train_idx]
        y_v = y_train[val_idx]
        cache.append((X_t, X_v, y_t, y_v))
        print(
            f"  Fold {fold_id}: train={len(y_t)}, validation={len(y_v)}, "
            f"features={X_t.shape[1]}, $20+ games={sum(y_v >= 20)}",
            flush=True
        )

    configs = generate_configs(args.trials, SEED)
    print(
        f"\nTuning {len(configs)} configurations x {args.folds} CV folds."
        " Saving scores after every completed configuration.", flush=True
    )
    rows = []
    fold_rows = []
    for trial_num, config in enumerate(configs, 1):
        started = time.perf_counter()
        per_fold = []
        print(
            f"\n[{trial_num}/{len(configs)}] "
            f"depth={config['max_depth']}, trees={config['n_estimators']}, "
            f"lr={config['learning_rate']}, "
            f"weights(<20/20+/40+)=(1/{config['weight20']:g}/{config['weight40']:g})",
            flush=True
        )
        status = "ok"
        error_message = ""
        try:
            for i, (X_t, X_v, y_t, y_v) in enumerate(cache, start=1):
                model = model_from_config(config, args.threads)
                weights = sample_weights(y_t, config["weight20"], config["weight40"])
                model.fit(X_t, y_t, sample_weight=weights)
                pred = model.predict(X_v)
                stat = metrics(y_v, pred)
                if not math.isfinite(stat["mae_20_plus"]):
                    raise ValueError("CV fold has no $20+ samples")
                per_fold.append(stat)
                fold_rows.append({"trial": trial_num, "fold": i, **stat})
                print(
                    f"  CV fold {i}: overall ${stat['overall_mae']:.2f}, "
                    f"$20+ ${stat['mae_20_plus']:.2f}, "
                    f"$40+ ${stat['mae_40_plus']:.2f}", flush=True
                )
        except Exception as exc:
            status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {error_message}", flush=True)

        trial_row = {"trial": trial_num, "status": status, **config}
        if status == "ok":
            # Fold mean, rather than just a single holdout's score.
            avg = lambda name: float(np.nanmean([f[name] for f in per_fold]))
            trial_row.update({
                "cv_overall_mae": avg("overall_mae"),
                "cv_mae_20_plus": avg("mae_20_plus"),
                "cv_mae_20_to_60": avg("mae_20_to_60"),
                "cv_mae_20_to_40": avg("mae_20_to_40"),
                "cv_mae_40_plus": avg("mae_40_plus"),
                "cv_bias_40_plus": avg("bias_40_plus"),
            })
            trial_row["cv_score"] = selection_score(
                trial_row["cv_overall_mae"], trial_row["cv_mae_20_plus"]
            )
            print(
                f"  => CV score {trial_row['cv_score']:.3f} "
                f"| overall ${trial_row['cv_overall_mae']:.2f} "
                f"| $20+ ${trial_row['cv_mae_20_plus']:.2f} "
                f"| $40+ ${trial_row['cv_mae_40_plus']:.2f}",
                flush=True
            )
        else:
            trial_row["error"] = error_message
        trial_row["seconds"] = round(time.perf_counter() - started, 1)
        rows.append(trial_row)
        save_results(rows, fold_rows, outdir)

    completed = [r for r in rows if r["status"] == "ok"]
    if not completed:
        raise RuntimeError("All tuning trials failed; inspect tuning_trials.csv")
    ranking = pd.DataFrame(completed).sort_values("cv_score")
    print("\nTOP CONFIGURATIONS BY CROSS-VALIDATED PREMIUM-FOCUSED SCORE:", flush=True)
    print(ranking[[
        "trial", "cv_score", "cv_overall_mae", "cv_mae_20_plus",
        "cv_mae_40_plus", "weight20", "weight40", "max_depth",
        "n_estimators", "learning_rate"
    ]].head(10).round(3).to_string(index=False), flush=True)

    best = min(completed, key=lambda r: r["cv_score"])
    print("\nFitting best configuration to full TRAIN split...", flush=True)
    prep = FeaturePreprocessor(list_columns)
    X_train = prep.fit_transform(train)
    X_validation = prep.transform(validation)
    model = model_from_config(best, args.threads)
    w = sample_weights(y_train, best["weight20"], best["weight40"])
    model.fit(X_train, y_train, sample_weight=w)
    prediction = model.predict(X_validation)
    final_stats = metrics(y_val, prediction)
    summary = pd.DataFrame([{
        "model": "Best CV weighted XGBoost",
        "trial": best["trial"],
        "cv_score": best["cv_score"],
        **final_stats,
    }])
    summary.to_csv(outdir / "validation_summary.csv", index=False)

    result = validation[[c for c in ["app_id", "name", "price"] if c in validation.columns]].copy()
    result["predicted"] = prediction
    result["error"] = result["predicted"] - result["price"]
    result["abs_error"] = result["error"].abs()
    result["price_range"] = pd.cut(
        result["price"], bins=BINS, labels=BAND_LABELS, right=False
    )
    bands = result.groupby("price_range", observed=True).agg(
        games=("price", "size"),
        actual_mean=("price", "mean"),
        predicted_mean=("predicted", "mean"),
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
    ).reset_index()
    bands.to_csv(outdir / "validation_by_price_range.csv", index=False)
    result.to_csv(outdir / "validation_predictions.csv", index=False)

    # A fully fitted candidate artifact; does NOT overwrite the app's model.
    package = {
        "model": model,
        "encoders": prep.encoders,
        "scaler": prep.scaler,
        "list_columns": list_columns,
        "numeric_columns": NUMERIC_COLUMNS,
        "weight20": best["weight20"],
        "weight40": best["weight40"],
        "cv_selection_score": best["cv_score"],
        "validation_metrics": final_stats,
    }
    joblib.dump(package, outdir / "tuned_xgboost_candidate.joblib")

    print("\nFINAL VALIDATION CHECK (never the TEST set):", flush=True)
    print(summary.round(3).to_string(index=False), flush=True)
    print("\nBY PRICE RANGE:", flush=True)
    print(bands.round(2).to_string(index=False), flush=True)
    print(f"\nAll results saved in: {outdir.resolve()}", flush=True)
    print("The TEST split remains untouched. Website model was not overwritten.", flush=True)


if __name__ == "__main__":
    main()
