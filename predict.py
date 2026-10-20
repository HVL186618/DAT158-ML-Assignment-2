import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

package = joblib.load("models/steam_price_model.pkl")


def encode_labels(game, field, encoder_name):
    encoder = package[encoder_name]

    value = game.get(field, [])

    if value is None:
        value = []

    if isinstance(value, str):
        value = value.split(",")

    # Match inputs case-insensitively to the labels
    # the encoder learned during training.
    known_labels = {
        label.casefold(): label
        for label in encoder.classes_
    }

    labels = [
        known_labels[item.strip().casefold()]
        for item in value
        if item.strip().casefold() in known_labels
    ]

    return encoder.transform([labels])


def predict_price(game):
    tags = encode_labels(
        game, "popular_tags", "tag_encoder"
    )

    genres = encode_labels(
        game, "genre", "genre_encoder"
    )

    details = encode_labels(
        game, "game_details", "details_encoder"
    )

    publishers = encode_labels(
        game, "publisher", "publisher_encoder"
    )

    developers = encode_labels(
        game, "developer", "developer_encoder"
    )

    numeric = pd.DataFrame([game])[
        package["numeric_columns"]
    ]

    numeric = package["scaler"].transform(numeric)

    # EXACTLY the same order as train_model.py
    features = hstack([
        tags,
        genres,
        details,
        publishers,
        developers,
        csr_matrix(numeric)
    ], format="csr")

    # Check for mismatched features
    expected = package["model"].n_features_in_
    actual = features.shape[1]

    assert actual == expected, (
        f"Expected {expected} features, got {actual}"
    )

    return float(package["model"].predict(features)[0])