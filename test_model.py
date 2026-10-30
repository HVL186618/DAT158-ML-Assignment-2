"""
Steam Game Price Predictor -- comparison of additional regression models.

Models use exactly the same 70/15/15 random_state=42 split as train_model.py.
The TEST partition is never used for training, selecting models, or reporting
scores. This script does NOT overwrite models/steam_price_model.pkl.

Results saved in:
    model_comparison_results.csv             summary metrics for each model
    model_comparison_price_ranges.csv        per-price-band errors and bias
    model_comparison_predictions.csv         validation-game predictions only

This is an exploratory model comparison on one validation split, NOT
hyperparameter optimization or cross-validation yet.
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import LinearSVR, SVR

SEED = 42
BANDS = [0, 10, 20, 40, float("inf")]
BAND_LABELS = ["$0-10", "$10-20", "$20-40", "$40+"]

# The warnings can contain *thousands* of individual new company names.
# Unseen labels being ignored on validation is normal with MultiLabelBinarizer.
warnings.filterwarnings(
    "ignore",
    message=r"unknown class\(es\).*will be ignored",
    category=UserWarning,
)


def parse_labels(series):
    """Match train_model.py: split comma-separated lists (including companies)."""
    return series.fillna("").astype(str).apply(
        lambda entry: [part.strip() for part in entry.split(",") if part.strip()]
    )


def make_features(train, validation):
    """Fit ALL preprocessing on training rows, transform validation only."""
    parts_train = []
    parts_validation = []

    # Same order/encoding as train_model.py.
    for column in [
        "popular_tags", "genre", "game_details", "publisher", "developer", "languages"
    ]:
        if column not in train.columns:
            if column == "languages":
                print("WARNING: languages missing in CSV; using no language features.", flush=True)
                continue
            raise ValueError(f"Missing required column: {column}")
        encoder = MultiLabelBinarizer(sparse_output=True)
        x_train = encoder.fit_transform(parse_labels(train[column]))
        x_validation = encoder.transform(parse_labels(validation[column]))
        parts_train.append(x_train)
        parts_validation.append(x_validation)
        print(f"  {column:15s}: {x_train.shape[1]:5d} features", flush=True)

    numeric_columns = ["achievements", "release_year"]
    numeric_train = train[numeric_columns].copy()
    numeric_validation = validation[numeric_columns].copy()

    if numeric_train.isna().any().any() or numeric_validation.isna().any().any():
        raise ValueError("Missing numeric features. Check your cleaned CSV.")

    scaler = StandardScaler()
    parts_train.append(csr_matrix(scaler.fit_transform(numeric_train)))
    parts_validation.append(csr_matrix(scaler.transform(numeric_validation)))

    X_train = hstack(parts_train, format="csr", dtype=np.float32)
    X_validation = hstack(parts_validation, format="csr", dtype=np.float32)
    print(f"Feature matrices: train={X_train.shape}, validation={X_validation.shape}", flush=True)
    return X_train, X_validation


def evaluate(name, predictions, y_val, elapsed):
    """Overall and premium-focused metrics; bias = prediction - actual."""
    actual = np.asarray(y_val, dtype=float)
    predicted = np.asarray(predictions, dtype=float).reshape(-1)
    if predicted.shape != actual.shape or not np.isfinite(predicted).all():
        raise ValueError(f"{name}: invalid predictions")

    error = predicted - actual
    abs_error = np.abs(error)

    def mae_for(mask):
        return float(abs_error[mask].mean()) if mask.any() else np.nan

    mask_20_plus = actual >= 20
    mask_20_60 = (actual >= 20) & (actual <= 60)
    mask_40_plus = actual >= 40
    mask_20_40 = (actual >= 20) & (actual < 40)

    summary = {
        "model": name,
        "overall_MAE": float(mean_absolute_error(actual, predicted)),
        "overall_RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE_20_plus": mae_for(mask_20_plus),
        "MAE_20_to_60": mae_for(mask_20_60),
        "within_5_20_to_60_pct": (
            float(100 * (abs_error[mask_20_60] <= 5).mean())
            if mask_20_60.any() else np.nan
        ),
        "MAE_20_to_40": mae_for(mask_20_40),
        "MAE_40_plus": mae_for(mask_40_plus),
        "bias_40_plus": (
            float(error[mask_40_plus].mean()) if mask_40_plus.any() else np.nan
        ),
        "seconds": round(elapsed, 1),
    }

    price_bands = pd.cut(
        actual, bins=BANDS, labels=BAND_LABELS, right=False
    )
    groups = pd.DataFrame({
        "price_range": price_bands,
        "actual": actual,
        "predicted": predicted,
        "error": error,
        "abs_error": abs_error,
    })
    per_band = groups.groupby("price_range", observed=True).agg(
        games=("actual", "size"),
        actual_mean=("actual", "mean"),
        predicted_mean=("predicted", "mean"),
        MAE=("abs_error", "mean"),
        bias=("error", "mean"),
    ).reset_index()
    per_band.insert(0, "model", name)
    return summary, per_band


def main():
    parser = argparse.ArgumentParser(description="Compare new Steam price regression models")
    parser.add_argument("--dataset", default="test.csv", help="Cleaned CSV (default: test.csv)")
    parser.add_argument("--quick", action="store_true", help="Run only lightweight smoke-test models")
    args = parser.parse_args()

    print(f"Reading {args.dataset}", flush=True)
    df = pd.read_csv(args.dataset)

    # Identical split strategy as train_model.py. Do NOT evaluate on test.
    train, remaining = train_test_split(df, test_size=0.30, random_state=SEED)
    validation, _untouched_test = train_test_split(
        remaining, test_size=0.50, random_state=SEED
    )
    del remaining, _untouched_test

    y_train = train["price"].to_numpy(dtype=float)
    y_validation = validation["price"].to_numpy(dtype=float)
    print(
        f"Train: {len(train)} | Validation: {len(validation)} | "
        f"TEST: {len(df) - len(train) - len(validation)} (NOT EVALUATED)",
        flush=True,
    )
    print(
        f"Validation samples priced $20+: {(y_validation >= 20).sum()}, "
        f"$40+: {(y_validation >= 40).sum()}", flush=True,
    )

    X_train, X_val = make_features(train, validation)
    summaries = []
    band_tables = []
    predictions_csv = validation[["name", "price"]].reset_index(drop=True).copy()

    def record(name, prediction, seconds):
        summary, bands = evaluate(name, prediction, y_validation, seconds)
        summaries.append(summary)
        band_tables.append(bands)
        predictions_csv[name] = np.asarray(prediction).reshape(-1)

        # Save after each model, so partial results survive an interruption.
        pd.DataFrame(summaries).to_csv("model_comparison_results.csv", index=False)
        pd.concat(band_tables, ignore_index=True).to_csv(
            "model_comparison_price_ranges.csv", index=False
        )
        predictions_csv.to_csv("model_comparison_predictions.csv", index=False)

        print(
            f"  Overall MAE ${summary['overall_MAE']:.2f} | "
            f"$20+ MAE ${summary['MAE_20_plus']:.2f} | "
            f"$40+ MAE ${summary['MAE_40_plus']:.2f} | "
            f"$40+ bias ${summary['bias_40_plus']:+.2f} | "
            f"{seconds:.1f}s\n",
            flush=True,
        )

    def run(name, model, features_train=X_train, features_val=X_val, sample_weight=None):
        print(f"\nTraining: {name}", flush=True)
        started = time.perf_counter()
        try:
            if sample_weight is None:
                model.fit(features_train, y_train)
            else:
                model.fit(features_train, y_train, sample_weight=sample_weight)
            pred = model.predict(features_val)
            record(name, pred, time.perf_counter() - started)
        except Exception as exc:
            print(f"  SKIPPED ({type(exc).__name__}: {exc})", flush=True)

    print("\n--- Baselines / new model candidates ---", flush=True)

    # Median baseline is intentionally a reference, not a newly tried algorithm.
    baseline = DummyRegressor(strategy="median")
    run("Median baseline", baseline, X_train[:, :1], X_val[:, :1])

    # The familiar Random Forest is included for a same-run comparison.
    run("Reference Random Forest", RandomForestRegressor(
        n_estimators=150, min_samples_leaf=2, random_state=SEED, n_jobs=-1
    ))

    # Linear SVR: different loss function from Ridge, sparse-friendly.
    run("Linear SVR", LinearSVR(
        C=2.0, epsilon=1.0, max_iter=6000, random_state=SEED
    ))

    # Extremely randomized trees: new tree-based competitor to Random Forest.
    run("Extra Trees", ExtraTreesRegressor(
        n_estimators=200, min_samples_leaf=2, random_state=SEED, n_jobs=-1
    ))

    if not args.quick:
        # scikit-learn boosting (different implementation from XGBoost).
        # Works directly with sparse data, though may run slowly.
        run("Sklearn Gradient Boosting", GradientBoostingRegressor(
            n_estimators=180, learning_rate=0.06, max_depth=3,
            min_samples_leaf=5, random_state=SEED
        ))

        # HistGradientBoosting and RBF SVR need dense input; NEVER densify the
        # full ~11,000-column dataset. Compress to 64 dense components instead.
        print("\nPreparing 64 SVD components for dense-only models...", flush=True)
        try:
            svd = TruncatedSVD(
                n_components=min(64, X_train.shape[1] - 1),
                n_iter=5, random_state=SEED
            )
            reduced_train = svd.fit_transform(X_train).astype(np.float32)
            reduced_val = svd.transform(X_val).astype(np.float32)
            print(f"  SVD matrix: {reduced_train.shape}", flush=True)

            run("HistGB (SVD 64)", HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.06, max_leaf_nodes=15,
                l2_regularization=3.0, random_state=SEED
            ), reduced_train, reduced_val)

            # Scale the reduced features for the RBF distance calculation.
            svd_scaler = StandardScaler()
            reduced_train_scaled = svd_scaler.fit_transform(reduced_train)
            reduced_val_scaled = svd_scaler.transform(reduced_val)
            run("RBF SVR (SVD 64)", SVR(
                kernel="rbf", C=10, epsilon=2.0, gamma="scale"
            ), reduced_train_scaled, reduced_val_scaled)
        except Exception as exc:
            print(f"  SVD models skipped ({type(exc).__name__}: {exc})", flush=True)

        # LightGBM and CatBoost: optional external gradient-boosting libraries.
        # These run if the packages are installed, otherwise print instructions.
        try:
            from lightgbm import LGBMRegressor
            run("LightGBM", LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=20,
                min_child_samples=15, colsample_bytree=0.85,
                random_state=SEED, n_jobs=-1, verbosity=-1
            ))
        except ImportError:
            print("\nLightGBM skipped: pip install lightgbm", flush=True)

        try:
            from catboost import CatBoostRegressor
            run("CatBoost", CatBoostRegressor(
                iterations=450, depth=6, learning_rate=0.05,
                loss_function="RMSE", random_seed=SEED,
                thread_count=-1, verbose=False
            ))
        except ImportError:
            print("\nCatBoost skipped: pip install catboost", flush=True)

        # Previously tried XGBoost algorithm, but a NEW weighted variant.
        # The weights are determined from TRAINING target values only.
        try:
            from xgboost import XGBRegressor
            premium_weights = np.where(
                y_train >= 40, 4.0, np.where(y_train >= 20, 2.0, 1.0)
            )
            run("Weighted XGBoost", XGBRegressor(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.8,
                objective="reg:squarederror", tree_method="hist",
                random_state=SEED, n_jobs=-1
            ), sample_weight=premium_weights)
        except ImportError:
            print("\nWeighted XGBoost skipped: pip install xgboost", flush=True)

    if not summaries:
        print("No models completed successfully.", flush=True)
        return

    full = pd.DataFrame(summaries)
    bands = pd.concat(band_tables, ignore_index=True)

    full.to_csv("model_comparison_results.csv", index=False)
    bands.to_csv("model_comparison_price_ranges.csv", index=False)
    predictions_csv.to_csv("model_comparison_predictions.csv", index=False)

    print("\n" + "=" * 98, flush=True)
    print("ALL MODELS - sorted by overall MAE (lower is better)", flush=True)
    print("=" * 98, flush=True)
    overview_columns = [
        "model", "overall_MAE", "MAE_20_plus", "MAE_20_to_60",
        "MAE_40_plus", "bias_40_plus", "within_5_20_to_60_pct", "seconds"
    ]
    print(full.sort_values("overall_MAE")[overview_columns].round(2).to_string(index=False), flush=True)

    print("\nPREMIUM FOCUS - sorted by $20-$60 MAE", flush=True)
    print(full.sort_values("MAE_20_to_60")[overview_columns].round(2).to_string(index=False), flush=True)

    print("\nPER-PRICE-BAND MAE (all rows from the same validation games):", flush=True)
    pivot = bands.pivot(index="model", columns="price_range", values="MAE")
    print(pivot.reindex(columns=BAND_LABELS).round(2).to_string(), flush=True)

    print("\nSaved: model_comparison_results.csv, model_comparison_price_ranges.csv, "
          "model_comparison_predictions.csv", flush=True)
    print("The held-out TEST set has NOT been evaluated.", flush=True)


if __name__ == "__main__":
    main()
