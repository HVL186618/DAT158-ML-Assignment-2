from functools import lru_cache
from pathlib import Path
import math

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack


MODEL_PATH = Path(__file__).resolve().parent / "model" / "steamduck20.pkl"

# IMPORTANT: Must match the hstack order in train_model.py exactly.
CATEGORICAL_FEATURES = (
    ("popular_tags", "tag_encoder"),
    ("genre", "genre_encoder"),
    ("game_details", "details_encoder"),
    ("publisher", "publisher_encoder"),
    ("developer", "developer_encoder"),
)


@lru_cache(maxsize=1)
def load_model_package():
    """Load the saved model and its fitted preprocessing once per Python process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL_PATH}. "
            "Run train_model.py to create models/steam_price_model.pkl, "
            "or copy your existing trained model there."
        )

    package = joblib.load(MODEL_PATH)
    required = {"model", "scaler", "numeric_columns"} | {
        encoder_key for _, encoder_key in CATEGORICAL_FEATURES
    }
    if not isinstance(package, dict) or not required.issubset(package):
        raise ValueError(
            "The .pkl file must be the model_package dictionary saved by train_model.py."
        )
    return package


def _labels(value):
    """Accept comma-separated text or a list, as used in predict_example.py."""
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        raise TypeError(f"Expected text or a list of labels, got {type(value).__name__}")

    return [
        part.strip()
        for item in values
        for part in str(item).split(",")
        if part.strip()
    ]


def _encoded_labels(raw_value, encoder):
    """Match the learned vocabulary without requiring users to type exact case.

    In clean_dataset.py the genre was NOT lowercased, whereas the other
    string features were. Mapping to fitted labels fixes inputs like 'action'
    when the model learned 'Action'. Unseen labels are ignored, as they are
    by MultiLabelBinarizer.transform during training/validation.
    """
    labels_by_case = {str(label).casefold(): label for label in encoder.classes_}
    result = []
    for label in _labels(raw_value):
        # Prefer exact spellings if different labels differ only by case.
        if label in encoder.classes_:
            result.append(label)
        elif label.casefold() in labels_by_case:
            result.append(labels_by_case[label.casefold()])
    return result


def build_feature_matrix(game, package):
    """Apply the *fitted* encoders and scaler in training column order."""
    category_parts = []
    for feature_name, encoder_name in CATEGORICAL_FEATURES:
        encoder = package[encoder_name]
        labels = _encoded_labels(game.get(feature_name), encoder)
        category_parts.append(encoder.transform([labels]))

    numeric_values = {}
    for column in package["numeric_columns"]:
        try:
            value = float(game[column])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Provide a valid number for '{column}'.") from exc
        if not math.isfinite(value):
            raise ValueError(f"'{column}' must be a finite number.")
        numeric_values[column] = value

    # DataFrame and column order must match the scaler fitted on train[numeric_columns].
    numeric_frame = pd.DataFrame([numeric_values], columns=package["numeric_columns"])
    scaled_numeric = package["scaler"].transform(numeric_frame)
    return hstack([*category_parts, csr_matrix(scaled_numeric)], format="csr")


def predict_price(game):
    """Return the predicted dollar price for a game described by `game`."""
    package = load_model_package()
    features = build_feature_matrix(game, package)
    return float(package["model"].predict(features)[0])
