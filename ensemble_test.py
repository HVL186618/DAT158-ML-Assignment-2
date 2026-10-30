"""courtesy of chatpgt

Compare ensembles for the Steam price prediction project.

An ensemble is a weighted average of predictions from different models.
This script tests mixtures of:
  - Normal Random Forest (your original 150-tree model)
  - Weighted Random Forest (price weights 1/2/4)
  - Your cross-validated, tuned WEIGHTED XGBoost

IMPORTANT:
  * Exactly the same 70/15/15 random_state=42 split as train_model.py.
  * 3-fold CV on TRAIN chooses the blend (preprocessing fitted per fold).
  * The held-out VALIDATION split is evaluated once at the end.
  * The final TEST split is never used for training, tuning or evaluation.
  * Website model files are never overwritten.

Files needed in the project directory:
  test.csv
  tuning_results/best_cv_configuration.json (created by tune_model.py)
  (or tuning_results/tuning_trials.csv if the JSON is missing)

Install: pip install pandas numpy scipy scikit-learn xgboost
Run:     python ensemble_test.py
Optional: python ensemble_test.py --dataset test.csv --folds 3 --threads 4

Results go into ensemble_results/ as CSV/JSON.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from xgboost import XGBRegressor

SEED = 42
LIST_COLUMNS = ["popular_tags", "genre", "game_details", "publisher", "developer", "languages"]
NUMERIC_COLUMNS = ["achievements", "release_year"]
MODEL_NAMES = ["Random Forest", "Weighted Random Forest", "Tuned Weighted XGBoost"]
BAND_LABELS = ["$0-10", "$10-20", "$20-40", "$40+"]
BINS = [0, 10, 20, 40, np.inf]

warnings.filterwarnings(
    "ignore", message=r"unknown class\(es\).*will be ignored", category=UserWarning
)


def parse_labels(series):
    # Same parsing approach as your train_model.py and tune_model.py.
    return series.fillna("").astype(str).apply(
        lambda s: [part.strip() for part in s.split(",") if part.strip()]
    )


class FeaturePreprocessor:
    """Fit encoders/scaler on training rows only, including in every CV fold."""

    def __init__(self, columns):
        self.columns = columns
        self.encoders = {}
        self.scaler = StandardScaler()

    def fit_transform(self, df):
        blocks = []
        for col in self.columns:
            enc = MultiLabelBinarizer(sparse_output=True)
            blocks.append(enc.fit_transform(parse_labels(df[col])))
            self.encoders[col] = enc
        numeric = df[NUMERIC_COLUMNS].astype(float)
        if numeric.isna().any().any():
            raise ValueError("Missing numeric data in achievements/release_year")
        blocks.append(csr_matrix(self.scaler.fit_transform(numeric)))
        return hstack(blocks, format="csr", dtype=np.float32)

    def transform(self, df):
        blocks = [
            self.encoders[col].transform(parse_labels(df[col]))
            for col in self.columns
        ]
        numeric = df[NUMERIC_COLUMNS].astype(float)
        if numeric.isna().any().any():
            raise ValueError("Missing numeric data in achievements/release_year")
        blocks.append(csr_matrix(self.scaler.transform(numeric)))
        return hstack(blocks, format="csr", dtype=np.float32)


def load_best_xgb_config(tuning_dir):
    json_file = tuning_dir / "best_cv_configuration.json"
    csv_file = tuning_dir / "tuning_trials.csv"
    if json_file.exists():
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = data.get("config", data)
        source = str(json_file)
    elif csv_file.exists():
        trials = pd.read_csv(csv_file)
        valid = trials.loc[trials["status"] == "ok"].sort_values("cv_score")
        if valid.empty:
            raise ValueError("No successful trials in tuning_trials.csv")
        config = valid.iloc[0].to_dict()
        source = str(csv_file)
    else:
        # Fallback: exact winning values from the user's 36-trial tune_model.py run.
        # The JSON is preferred because it reflects their own latest tuning run.
        config = {
            "n_estimators": 700, "max_depth": 5, "learning_rate": 0.04,
            "min_child_weight": 2, "subsample": 0.85,
            "colsample_bytree": 0.65, "reg_alpha": 1.0,
            "reg_lambda": 1.0, "gamma": 2.0,
            "weight20": 3.0, "weight40": 6.0,
        }
        source = "FALLBACK: trial 35 values from previous tuning results"
    keys = (
        "n_estimators", "max_depth", "learning_rate", "min_child_weight",
        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
        "gamma", "weight20", "weight40"
    )
    for key in keys:
        if key not in config:
            raise ValueError(f"Tuned XGBoost config lacks {key}")
    return config, source


def price_weights(y, w20, w40):
    return np.where(y >= 40, w40, np.where(y >= 20, w20, 1.0))


def train_three(X_train, y_train, X_predict, config, threads):
    """Train all three models on the same data, return their predictions."""
    models = [
        RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2,
            random_state=SEED, n_jobs=threads,
        ),
        RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2,
            random_state=SEED, n_jobs=threads,
        ),
        XGBRegressor(
            objective="reg:squarederror", tree_method="hist",
            random_state=SEED, n_jobs=threads, verbosity=0,
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            learning_rate=float(config["learning_rate"]),
            min_child_weight=float(config["min_child_weight"]),
            subsample=float(config["subsample"]),
            colsample_bytree=float(config["colsample_bytree"]),
            reg_alpha=float(config["reg_alpha"]),
            reg_lambda=float(config["reg_lambda"]),
            gamma=float(config["gamma"]),
        ),
    ]
    weights = [
        None,
        price_weights(y_train, 2.0, 4.0),
        price_weights(y_train, float(config["weight20"]), float(config["weight40"])),
    ]
    predictions = []
    for name, model, w in zip(MODEL_NAMES, models, weights):
        print(f"    Training {name}...", flush=True)
        if w is None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, sample_weight=w)
        predictions.append(model.predict(X_predict))
    return np.column_stack(predictions)


def generate_blends():
    """Base models + two-model blends + coarse three-model blends."""
    mixes = {}
    for i, model in enumerate(MODEL_NAMES):
        w = [0.0, 0.0, 0.0]
        w[i] = 1.0
        mixes[model] = w

    for forest_index, prefix in [(0, "RF+XGB"), (1, "WeightedRF+XGB")]:
        for tenths in range(1, 10):
            xgb_weight = tenths / 10
            w = [0.0, 0.0, xgb_weight]
            w[forest_index] = 1 - xgb_weight
            mixes[f"{prefix} (XGB {tenths*10}%)"] = w

    for rf_parts in range(0, 6):
        for weighted_parts in range(0, 6 - rf_parts):
            xgb_parts = 5 - rf_parts - weighted_parts
            if min(rf_parts, weighted_parts, xgb_parts) == 0:
                continue  # Two-model blends are already included more finely.
            mixes[
                f"3-way (RF {rf_parts*20}%, weightedRF {weighted_parts*20}%, "
                f"XGB {xgb_parts*20}%)"
            ] = [rf_parts / 5, weighted_parts / 5, xgb_parts / 5]
    return mixes


def metrics(y, predicted):
    y = np.asarray(y, dtype=float)
    p = np.asarray(predicted, dtype=float)
    err = p - y
    absolute = np.abs(err)

    def avg(arr):
        return float(arr.mean()) if len(arr) else np.nan

    m20 = y >= 20
    m40 = y >= 40
    m2040 = (y >= 20) & (y < 40)
    m2060 = (y >= 20) & (y <= 60)
    overall = avg(absolute)
    premium = avg(absolute[m20])
    return {
        "overall_mae": overall,
        "mae_20_plus": premium,
        "mae_20_to_60": avg(absolute[m2060]),
        "mae_20_to_40": avg(absolute[m2040]),
        "mae_40_plus": avg(absolute[m40]),
        "bias_20_plus": avg(err[m20]),
        "bias_40_plus": avg(err[m40]),
        "within_5_20_to_60_pct": avg((absolute[m2060] <= 5) * 100),
        "n_20_plus": int(m20.sum()),
        "n_40_plus": int(m40.sum()),
        "score": 0.30 * overall + 0.70 * premium,
    }


def cv_metrics(y, preds, folds, mixes):
    """Compute one set of out-of-fold metrics on TRAIN for every blend."""
    results = []
    for name, weights in mixes.items():
        predictions = preds @ np.asarray(weights)
        record = {"blend": name, "rf_weight": weights[0],
                  "weighted_rf_weight": weights[1], "xgb_weight": weights[2]}
        record.update(metrics(y, predictions))
        # Include the average score across folds as a stability diagnostic.
        record["mean_fold_score"] = float(np.mean([
            metrics(y[indices], predictions[indices])["score"] for indices in folds
        ]))
        results.append(record)
    return pd.DataFrame(results).sort_values("score").reset_index(drop=True)


def price_ranges(y, preds, model_name):
    data = pd.DataFrame({"price": y, "predicted": preds})
    data["abs_error"] = (data["predicted"] - data["price"]).abs()
    data["error"] = data["predicted"] - data["price"]
    data["price_range"] = pd.cut(data["price"], bins=BINS, labels=BAND_LABELS, right=False)
    bands = data.groupby("price_range", observed=True).agg(
        games=("price", "size"), actual_mean=("price", "mean"),
        predicted_mean=("predicted", "mean"),
        mae=("abs_error", "mean"), bias=("error", "mean"),
    ).reset_index()
    bands.insert(0, "model", model_name)
    return bands


def main():
    parser = argparse.ArgumentParser(description="Steam price model blending experiment")
    parser.add_argument("--dataset", default="test.csv")
    parser.add_argument("--tuning-dir", default="tuning_results")
    parser.add_argument("--outdir", default="ensemble_results")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.folds < 2 or args.threads < 1:
        parser.error("Use at least 2 folds and at least 1 thread")

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    config, source = load_best_xgb_config(Path(args.tuning_dir))
    print(f"Using XGBoost hyperparameters from: {source}", flush=True)
    print(f"Tuned price weights: 1 / {float(config['weight20']):g} / {float(config['weight40']):g}", flush=True)

    df = pd.read_csv(args.dataset)
    columns = [col for col in LIST_COLUMNS if col in df.columns]
    required = set(LIST_COLUMNS[:-1] + NUMERIC_COLUMNS + ["price"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns in {args.dataset}: {missing}")
    if "languages" not in df.columns:
        print("WARNING: No languages column. Use the newer test.csv to reproduce your latest results.", flush=True)

    train, rem = train_test_split(df, test_size=.30, random_state=SEED)
    validation, final_test = train_test_split(rem, test_size=.50, random_state=SEED)
    print(f"TRAIN {len(train)} | VALIDATION {len(validation)} | UNTOUCHED TEST {len(final_test)}", flush=True)
    del df, rem, final_test
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    y_train = train["price"].to_numpy(dtype=float)
    y_validation = validation["price"].to_numpy(dtype=float)
    if not np.isfinite(y_train).all() or not np.isfinite(y_validation).all():
        raise ValueError("Missing/nonfinite price labels")

    mixes = generate_blends()
    cv = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    pred_cv = np.full((len(train), 3), np.nan)
    fold_indices = []
    print(f"\nCross-validation: {args.folds} folds, 3 models per fold, {len(mixes)} blends.", flush=True)
    for fold, (tr_idx, vl_idx) in enumerate(cv.split(train), 1):
        print(f"\nCV fold {fold}/{args.folds}...", flush=True)
        fold_train, fold_val = train.iloc[tr_idx], train.iloc[vl_idx]
        prep = FeaturePreprocessor(columns)
        X_train = prep.fit_transform(fold_train)
        X_val = prep.transform(fold_val)
        pred_cv[vl_idx, :] = train_three(
            X_train, y_train[tr_idx], X_val, config, args.threads,
        )
        fold_indices.append(vl_idx)
    if not np.isfinite(pred_cv).all():
        raise RuntimeError("Not all training rows received out-of-fold predictions")

    cv_table = cv_metrics(y_train, pred_cv, fold_indices, mixes)
    cv_table.to_csv(outdir / "ensemble_cv_results.csv", index=False)
    winner = cv_table.iloc[0]
    winner_name = winner["blend"]
    print("\nTOP BLENDS CHOSEN BY TRAIN-ONLY CROSS-VALIDATION (lower is better):", flush=True)
    print(cv_table[["blend", "score", "overall_mae", "mae_20_plus", "mae_40_plus"]]
          .head(12).round(2).to_string(index=False), flush=True)
    print(f"\nChosen blend: {winner_name}", flush=True)

    # Now and only now: train once on all TRAIN rows and evaluate VALIDATION.
    print("\nFitting 3 models on all TRAIN rows to evaluate held-out VALIDATION...", flush=True)
    prep = FeaturePreprocessor(columns)
    X_train = prep.fit_transform(train)
    X_val = prep.transform(validation)
    pred_val = train_three(X_train, y_train, X_val, config, args.threads)

    rows = []
    selected_bands = []
    for name, weights in mixes.items():
        predictions = pred_val @ np.asarray(weights)
        row = {"blend": name, "rf_weight": weights[0],
               "weighted_rf_weight": weights[1], "xgb_weight": weights[2]}
        row.update(metrics(y_validation, predictions))
        rows.append(row)
        if name in {winner_name, *MODEL_NAMES}:
            selected_bands.append(price_ranges(y_validation, predictions, name))
    val_table = pd.DataFrame(rows).sort_values("score")
    val_table.to_csv(outdir / "ensemble_validation_results.csv", index=False)
    pd.concat(selected_bands, ignore_index=True).to_csv(
        outdir / "ensemble_validation_by_price_range.csv", index=False
    )

    choice = mixes[winner_name]
    selected = pred_val @ np.asarray(choice)
    game_cols = [col for col in ["app_id", "name", "price"] if col in validation.columns]
    individual = validation[game_cols].copy()
    individual["rf_prediction"] = pred_val[:, 0]
    individual["weighted_rf_prediction"] = pred_val[:, 1]
    individual["tuned_xgb_prediction"] = pred_val[:, 2]
    individual["chosen_blend_prediction"] = selected
    individual["chosen_blend_error"] = selected - y_validation
    individual["price_range"] = pd.cut(y_validation, bins=BINS, labels=BAND_LABELS, right=False)
    individual.to_csv(outdir / "ensemble_validation_predictions.csv", index=False)
    with open(outdir / "chosen_blend.json", "w", encoding="utf-8") as f:
        json.dump({"chosen_on": "train-only out-of-fold cross-validation",
                   "blend": winner_name,
                   "weights": dict(zip(MODEL_NAMES, [float(w) for w in choice])),
                   "tuned_xgb_config_source": source}, f, indent=2)

    print("\nHELD-OUT VALIDATION — MAIN MODELS AND CV-CHOSEN BLEND:", flush=True)
    subset = val_table[val_table["blend"].isin({winner_name, *MODEL_NAMES})]
    print(subset[["blend", "overall_mae", "mae_20_plus", "mae_20_to_60", "mae_40_plus",
                  "bias_40_plus", "within_5_20_to_60_pct"]]
          .round(2).to_string(index=False), flush=True)
    print("\nVALIDATION PRICE BANDS FOR CV-CHOSEN BLEND:", flush=True)
    print(price_ranges(y_validation, selected, winner_name).round(2).to_string(index=False), flush=True)
    print(f"\nSaved CSV and JSON reports to: {outdir.resolve()}", flush=True)
    print("Final TEST set never used. No website model files changed.", flush=True)
    print("NOTE: This is an experiment; select your final model before touching the TEST set.", flush=True)


if __name__ == "__main__":
    main()
