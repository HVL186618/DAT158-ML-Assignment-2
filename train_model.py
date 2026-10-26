"""
a big mess of models for testing


"""
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

dataset = "test.csv"
# dataset = "steam_cleaned_experimental_cluster_filtered.csv"

# skip = ["forest", "wforest"]

# ---
"script:"

df = pd.read_csv(dataset)

train, remaining = train_test_split(
    df,
    test_size=.3,
    random_state=42 # the answer to the ultimate question of life, the universe, and everything
)

validation, test = train_test_split(
    remaining,
    test_size=.5,
    random_state=42
)

print(f"Training: {len(train)}")
print(f"Validation: {len(validation)}")
print(f"Testing: {len(test)}")

"""
baseline dummy
"""
baseline = DummyRegressor(strategy="median")

X_train = [[0]] * len(train)
X_validation = [[0]] * len(validation)

y_train = train["price"]
y_validation = validation["price"]

baseline.fit(X_train, y_train)

predictions = baseline.predict(X_validation)

mae = mean_absolute_error(
    y_validation,
    predictions
)

# print(
#     df[df["price"] >= 40]["price"]
#     .value_counts()
#     .head(25)
# )
#
# print(
#     df["price"].describe(
#         percentiles=[.90, .95, .99]
#     )
# )

print(f"Baseline MAE: ${mae:.2f}")

# ---
"""
features and stuff

feature:        type:                   preprocessing
popular_tags    multiple categories     multi-hot encoding
genre           multiple categories     multi-hot encoding
game_details    multiple categories     multi-hot encoding
achievements    numeric                 scaling for ridge
release_year    numeric                 scaling for ridge
"""

from sklearn.preprocessing import MultiLabelBinarizer # encode list data as binary labels

def parse_tags(series): # converts comma separated strings into python lists
    return series.fillna("").apply(
        lambda value: [
            tag.strip()
            for tag in value.split(",")
            if tag.strip()
        ]
    )

train_tags = parse_tags(train["popular_tags"])
validation_tags = parse_tags(validation["popular_tags"])

# learn what tags exist in the training dataset
tag_encoder = MultiLabelBinarizer(sparse_output=True)

X_train_tags = tag_encoder.fit_transform(train_tags)

# now validation tags
X_validation_tags = tag_encoder.transform(validation_tags)

print(f"Training shape: {X_train_tags.shape}")
print(f"Validation shape: {X_validation_tags.shape}")
print(f"Number of tags: {len(tag_encoder.classes_)}")

# ---
"""
genre, game details, and languages
"""

genre_encoder = MultiLabelBinarizer(sparse_output=True)
X_train_genres = genre_encoder.fit_transform(
    parse_tags(train["genre"]) # reusing tags parsing for genres
)
X_validation_genres = genre_encoder.transform(
    parse_tags(validation["genre"])
)

# game details
details_encoder = MultiLabelBinarizer(sparse_output=True)

X_train_details = details_encoder.fit_transform(
    parse_tags(train["game_details"])
)
X_validation_details = details_encoder.transform(
    parse_tags(validation["game_details"])
)

# languages
languages_encoder = MultiLabelBinarizer(sparse_output=True)

X_train_languages = languages_encoder.fit_transform(
    parse_tags(train["languages"])
)
X_validation_languages = languages_encoder.transform(
    parse_tags(validation["languages"])
)

print(f"Genres: {len(genre_encoder.classes_)}")
print(f"Game details: {len(details_encoder.classes_)}")
print(f"Languages: {len(languages_encoder.classes_)}")

# ---
"""
prepare numerical data
"""
from sklearn.preprocessing import StandardScaler

numeric_columns = "achievements release_year".split(" ")

scaler = StandardScaler()

X_train_numeric = scaler.fit_transform(
    train[numeric_columns]
)
X_validation_numeric = scaler.transform(
    validation[numeric_columns]
)

# ---
"""
publisher and developer info
"""

developer_encoder = MultiLabelBinarizer(sparse_output=True)

X_train_developer = developer_encoder.fit_transform(
    parse_tags(train["developer"])
)
X_validation_developer = developer_encoder.transform(
    parse_tags(validation["developer"])
)

publisher_encoder = MultiLabelBinarizer(sparse_output=True)

X_train_publisher = publisher_encoder.fit_transform(
    parse_tags(train["publisher"]) # reusing tags parsing for genres
)
X_validation_publisher = publisher_encoder.transform(
    parse_tags(validation["publisher"])
)

print(f"Developers: {len(developer_encoder.classes_)}")
print(f"Publishers: {len(publisher_encoder.classes_)}")

# ---
"""
combine features!
"""
from scipy.sparse import hstack, csr_matrix

X_train_features = hstack([
    X_train_tags,
    X_train_genres,
    X_train_details,
    X_train_publisher,
    X_train_developer,
    X_train_languages,
    csr_matrix(X_train_numeric)
], format="csr")

X_validation_features = hstack([
    X_validation_tags,
    X_validation_genres,
    X_validation_details,
    X_validation_publisher,
    X_validation_developer,
    X_validation_languages,
    csr_matrix(X_validation_numeric)
], format="csr")

print(f"Combined training: {X_train_features.shape}")
print(f"Combined validation: {X_validation_features.shape}")

# ---
"""
TRAINING YAY
"""
from xgboost import XGBRegressor

boost = XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

boost.fit(X_train_features, y_train)

boost_predictions = boost.predict(X_validation_features)

print(
    f"XGBoost MAE:{mean_absolute_error(y_validation, boost_predictions):.2f}"
)

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0, solver="lsqr")
ridge.fit(X_train_features, y_train)
ridge_predictions = ridge.predict(X_validation_features)

ridge_mae = mean_absolute_error(
    y_validation,
    ridge_predictions
)

print(f"Baseline MAE: ${mae:.2f}")
print(f"Ridge MAE: ${ridge_mae:.2f}")

"""
nonlinear shenanigans
"""
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(
    n_estimators=150,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=1
)
forest.fit(X_train_features, y_train)
forest_predictions = forest.predict(X_validation_features)

forest_mae = mean_absolute_error(
    y_validation,
    forest_predictions
)

print(f"Random Forest MAE: ${forest_mae:.2f}")

"""
weighted forest might do better at expensive games
"""
import numpy as np

# Give expensive games more influence
weights = np.where(
    y_train >= 40, 4,
    np.where(y_train >= 20, 2, 1)
)

weighted_forest = RandomForestRegressor(
    n_estimators=150,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=1
)

weighted_forest.fit(
    X_train_features,
    y_train,
    sample_weight=weights
)

weighted_predictions = weighted_forest.predict(
    X_validation_features
)

weighted_mae = mean_absolute_error(
    y_validation,
    weighted_predictions
)

print(f"Normal Forest MAE: ${forest_mae:.2f}")
print(f"Weighted Forest MAE: ${weighted_mae:.2f}")

# ==========================
# Error investigation
# ==========================
import matplotlib.pyplot as plt

# Compare actual and predicted prices
results = validation[["name", "price"]].copy()
results["predicted"] = forest_predictions

# Calculate errors
results["error"] = results["predicted"] - results["price"]
results["abs_error"] = results["error"].abs()

# Group games into price ranges
results["price_range"] = pd.cut(
    results["price"],
    bins=[0, 10, 20, 40, float("inf")],
    labels=["$0–10", "$10–20", "$20–40", "$40+"],
    right=False
)

# Calculate statistics for each range
stats = results.groupby("price_range", observed=True).agg(
    games=("price", "count"),
    actual_mean=("price", "mean"),
    predicted_mean=("predicted", "mean"),
    mae=("abs_error", "mean"),
    bias=("error", "mean")
)

print(stats.round(2))

# Visualize the errors
stats["mae"].plot(kind="bar", figsize=(9, 5))

plt.xlabel("Actual Price Range")
plt.ylabel("Mean Absolute Error ($)")
plt.title("Random Forest Prediction Error by Price Range")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

worst = results.nlargest(15, "abs_error")

# print("\nWorst predictions:")
# print(
#     worst[[
#         "name", "price", "predicted", "error"
#     ]].round(2).to_string(index=False)
# )

# train
train_predictions = forest.predict(X_train_features)

train_mae = mean_absolute_error(
    y_train,
    train_predictions
)

print(f"Training MAE: ${train_mae:.2f}")
print(f"Validation MAE: ${forest_mae:.2f}")


"""



"""


# Recreate our best Random Forest (leaf size 2)
forest = RandomForestRegressor(
    n_estimators=150,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

forest.fit(X_train_features, y_train)

results = validation[["name", "price"]].copy()
results["predicted"] = forest.predict(X_validation_features)
results["abs_error"] = (
    results["predicted"] - results["price"]
).abs()

# Exclude very cheap and extremely expensive games
normal_games = results[
    (results["price"] >= 10) &
    (results["price"] <= 60)
]

# Find reasonably accurate predictions
good_predictions = normal_games[
    normal_games["abs_error"] <= 5
]

# print("\nGood predictions:")
# print(
#     good_predictions.sample(
#         n=min(20, len(good_predictions)),
#         random_state=42
#     )[["name", "price", "predicted", "abs_error"]]
#     .round(2)
#     .to_string(index=False)
# )

print(
    f"\n{len(good_predictions)} out of "
    f"{len(normal_games)} games predicted within $5"
)

print(
    f"Accuracy rate: "
    f"{len(good_predictions) / len(normal_games) * 100:.1f}%"
)


print("---\n\n\n\n")
comparison = validation[["price"]].copy()

comparison["normal"] = forest_predictions
comparison["weighted"] = weighted_predictions

comparison["price_range"] = pd.cut(
    comparison["price"],
    bins=[0, 10, 20, 40, float("inf")],
    labels=["$0–10", "$10–20", "$20–40", "$40+"],
    right=False
)

for model in ["normal", "weighted"]:
    comparison[f"{model}_error"] = (
        comparison[model] - comparison["price"]
    )
    comparison[f"{model}_mae"] = (
        comparison[f"{model}_error"].abs()
    )

stats = comparison.groupby(
    "price_range", observed=True
)[[
    "normal_mae",
    "weighted_mae",
    "normal_error",
    "weighted_error"
]].mean()

print(stats.round(2))

"""
xgboost stats
"""
boost_results = validation[["price"]].copy()

boost_results["predicted"] = boost_predictions

boost_results["error"] = (
    boost_results["predicted"] - boost_results["price"]
)

boost_results["abs_error"] = boost_results["error"].abs()

boost_results["price_range"] = pd.cut(
    boost_results["price"],
    bins=[0, 10, 20, 40, float("inf")],
    labels=["$0–10", "$10–20", "$20–40", "$40+"],
    right=False
)

print(
    boost_results.groupby("price_range", observed=True).agg(
        games=("price", "count"),
        mae=("abs_error", "mean"),
        bias=("error", "mean")
    ).round(2)
)

# ---
"""
pickle model
"""
import joblib
import os



# Recreate our chosen Random Forest
final_forest = RandomForestRegressor(
    n_estimators=150,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

final_forest.fit(X_train_features, y_train, sample_weight=weights)

# Package model and preprocessing together
model_package = {
    "model": final_forest,

    "tag_encoder": tag_encoder,
    "genre_encoder": genre_encoder,
    "details_encoder": details_encoder,

    "publisher_encoder": publisher_encoder,
    "developer_encoder": developer_encoder,
    "languages_encoder": languages_encoder,

    "scaler": scaler,
    "numeric_columns": numeric_columns
}

os.makedirs("models", exist_ok=True)

joblib.dump(model_package, "models/steam_price_model.pkl")

print("Model saved!")

import joblib

joblib.dump(scaler, "scaler.joblib")