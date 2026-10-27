from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# annoying warnings
warnings.filterwarnings("ignore", message=r"unknown class\(es\).*will be ignored", category=UserWarning)

# same dataset as steamduck20
df = pd.read_csv("test.csv")

# same split as train_final.py.
train, remaining = train_test_split(df, test_size=.3, random_state=42)
validation, test = train_test_split(remaining, test_size=.5, random_state=42)

# because models/ is misbehaving on github
model_path = Path("model/steamduck20.pkl")
if not model_path.exists():
    model_path = Path("models/steamduck20.pkl")

package = joblib.load(model_path)
model = package["model"]


def parse_tags(series):
    return series.fillna("").astype(str).apply(
        lambda value: [tag.strip() for tag in value.split(",") if tag.strip()]
    )

def prepare_features(games):
    # This order must be identical to the order in train_final.py.
    tags = package["tag_encoder"].transform(parse_tags(games["popular_tags"]))
    genres = package["genre_encoder"].transform(parse_tags(games["genre"]))
    details = package["details_encoder"].transform(parse_tags(games["game_details"]))
    publishers = package["publisher_encoder"].transform(parse_tags(games["publisher"]))
    developers = package["developer_encoder"].transform(parse_tags(games["developer"]))
    languages = package["languages_encoder"].transform(parse_tags(games["languages"]))

    numeric_columns = package["numeric_columns"]
    numeric = package["scaler"].transform(games[numeric_columns])

    return hstack([
        tags, genres, details, publishers, developers, languages,
        csr_matrix(numeric)
    ], format="csr", dtype=np.float32)

# Sanity check: does this CSV reproduce the saved validation result?
# This helps catch accidental changes to the dataset or its row order.
if "validation_metrics" in package and "overall_mae" in package["validation_metrics"]:
    saved_mae = package["validation_metrics"]["overall_mae"]
    validation_predictions = model.predict(prepare_features(validation))
    validation_mae = mean_absolute_error(validation["price"], validation_predictions)
    if abs(validation_mae - saved_mae) > 0.05:
        raise ValueError(
            f"Validation doesn't match: saved MAE ${saved_mae:.2f}, "
            f"current MAE ${validation_mae:.2f}. "
            "Check that test.csv and steamduck20.pkl are the original versions."
        )
    print(f"Validation sanity check passed: ${validation_mae:.2f}")

print(f"Evaluating {len(test)} previously untouched TEST games...")


# ---
# final predictions :D
predictions = model.predict(prepare_features(test))

results = test[["name", "price"]].copy()
results["predicted"] = predictions
results["error"] = results["predicted"] - results["price"]
results["abs_error"] = results["error"].abs()
results["price_range"] = pd.cut(
    results["price"],
    bins=[0, 10, 20, 40, float("inf")],
    labels=["$0-10", "$10-20", "$20-40", "$40+"],
    right=False
)

# Same metrics we tracked during model selection.
premium = results["price"] >= 20
expensive = results["price"] >= 40
mid = (results["price"] >= 20) & (results["price"] < 40)
focus = (results["price"] >= 20) & (results["price"] <= 60)

print("\nSTEAMDUCK20 — FINAL TEST RESULTS")
print(f"Overall MAE:          ${results['abs_error'].mean():.2f}")
print(f"$20+ MAE:             ${results.loc[premium, 'abs_error'].mean():.2f}")
print(f"$20-40 MAE:           ${results.loc[mid, 'abs_error'].mean():.2f}")
print(f"$40+ MAE:             ${results.loc[expensive, 'abs_error'].mean():.2f}")
print(f"$40+ bias:            ${results.loc[expensive, 'error'].mean():.2f}")
print(f"$20-60 MAE:           ${results.loc[focus, 'abs_error'].mean():.2f}")
print(f"$20-60 within $5:     {(results.loc[focus, 'abs_error'] <= 5).mean() * 100:.1f}%")

by_range = results.groupby("price_range", observed=True).agg(
    games=("price", "count"),
    actual_mean=("price", "mean"),
    predicted_mean=("predicted", "mean"),
    mae=("abs_error", "mean"),
    bias=("error", "mean")
)
print("\nBY PRICE RANGE:")
print(by_range.round(2).to_string())
