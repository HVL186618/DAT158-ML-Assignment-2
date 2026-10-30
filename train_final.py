"""train steamduck20"""
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import mean_absolute_error

from scipy.sparse import hstack, csr_matrix
from xgboost import XGBRegressor


# our first final model contender: tuned weighted XGBoost (trial 35)
dataset = "test.csv"  # original cleaned dataset

df = pd.read_csv(dataset)

# same split as train_model.py
train, remaining = train_test_split(
    df, test_size=.3, random_state=42
)
validation, test = train_test_split(
    remaining, test_size=.5, random_state=42
)

print(f"Training: {len(train)}")
print(f"Validation: {len(validation)}")
print(f"Testing (untouched): {len(test)}")

y_train = train["price"]
y_validation = validation["price"]


# --- features and stuff ---

def parse_tags(series):
    return series.fillna("").apply(
        lambda value: [tag.strip() for tag in value.split(",") if tag.strip()]
    )


# encode each list of tags / categories
# these are fitted only on training data
tag_encoder = MultiLabelBinarizer(sparse_output=True)
X_train_tags = tag_encoder.fit_transform(parse_tags(train["popular_tags"]))
X_validation_tags = tag_encoder.transform(parse_tags(validation["popular_tags"]))

genre_encoder = MultiLabelBinarizer(sparse_output=True)
X_train_genres = genre_encoder.fit_transform(parse_tags(train["genre"]))
X_validation_genres = genre_encoder.transform(parse_tags(validation["genre"]))

details_encoder = MultiLabelBinarizer(sparse_output=True)
X_train_details = details_encoder.fit_transform(parse_tags(train["game_details"]))
X_validation_details = details_encoder.transform(parse_tags(validation["game_details"]))

publisher_encoder = MultiLabelBinarizer(sparse_output=True)
X_train_publisher = publisher_encoder.fit_transform(parse_tags(train["publisher"]))
X_validation_publisher = publisher_encoder.transform(parse_tags(validation["publisher"]))

developer_encoder = MultiLabelBinarizer(sparse_output=True)
X_train_developer = developer_encoder.fit_transform(parse_tags(train["developer"]))
X_validation_developer = developer_encoder.transform(parse_tags(validation["developer"]))

languages_encoder = MultiLabelBinarizer(sparse_output=True)
X_train_languages = languages_encoder.fit_transform(parse_tags(train["languages"]))
X_validation_languages = languages_encoder.transform(parse_tags(validation["languages"]))

# numerical features
numeric_columns = ["achievements", "release_year"]
scaler = StandardScaler()
X_train_numeric = scaler.fit_transform(train[numeric_columns])
X_validation_numeric = scaler.transform(validation[numeric_columns])

# combine everything
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

print(f"Training shape: {X_train_features.shape}")
print(f"Validation shape: {X_validation_features.shape}")


# --- training yay ---

# expensive games get more weight/influence during training
weights = np.where(
    y_train >= 40, 6,
    np.where(y_train >= 20, 3, 1)
)

# best settings from tune_model.py, trial 35
boost = XGBRegressor(
    n_estimators=700,
    max_depth=5,
    learning_rate=0.04,
    min_child_weight=2,
    subsample=0.85,
    colsample_bytree=0.65,
    reg_lambda=1.0,
    reg_alpha=1.0,
    gamma=2.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

boost.fit(X_train_features, y_train, sample_weight=weights)
predictions = boost.predict(X_validation_features)


# --- validation results ---

errors = np.abs(predictions - y_validation.to_numpy())
prices = y_validation.to_numpy()

premium = prices >= 20
mid_price = (prices >= 20) & (prices < 40)
expensive = prices >= 40
focus = (prices >= 20) & (prices <= 60)

print(f"\nOverall MAE: ${mean_absolute_error(prices, predictions):.2f}")
print(f"$20+ MAE: ${errors[premium].mean():.2f}")
print(f"$20-40 MAE: ${errors[mid_price].mean():.2f}")
print(f"$40+ MAE: ${errors[expensive].mean():.2f}")
print(f"$20-60 MAE: ${errors[focus].mean():.2f}")
print(f"$40+ bias: ${(predictions[expensive] - prices[expensive]).mean():.2f}")
print(f"$20-60 within $5: {(errors[focus] <= 5).mean() * 100:.1f}%")


# --- save the him ---
model_package = {
    "model": boost,
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
joblib.dump(model_package, "model/steamduck20.pkl")

print("\nContender saved: model/steamduck20.pkl") # duckomotive
