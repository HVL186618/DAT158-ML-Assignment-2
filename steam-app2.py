"""
>streamlit run steam-app.py
"""

import streamlit as st

from predict import MODEL_PATH, predict_price


st.set_page_config(page_title="Steam Price Predictor", page_icon="🎮")
st.title("🎮 Steam Price Predictor")
st.write("HVL ML Assignment 2 Project")
st.write("Enter a game's details to estimate its Steam price in USD.")

# The training script packages the fitted model, encoders and scaler
# together in ONE file under the 'models' directory (plural).
if not MODEL_PATH.is_file():
    st.error(
        "Model file not found: `models/steam_price_model.pkl`. "
        "Copy your trained model package into the `models` folder "
        "next to `steam-app.py` and `predict.py`."
    )
    st.stop()

with st.form("steam_game_prediction"):
    col1, col2 = st.columns(2)
    with col1:
        publisher = st.text_input(
            "Publisher", placeholder="Annapurna Interactive"
        )
    with col2:
        developer = st.text_input(
            "Developer", placeholder="Mobius Digital"
        )

    popular_tags = st.text_input(
        "Popular tags",
        placeholder="Exploration, Space, Mystery",
        help="Separate multiple tags with commas.",
    )
    genre = st.text_input(
        "Genre(s)",
        placeholder="Adventure, Indie",
        help="Separate multiple genres with commas.",
    )
    game_details = st.text_input(
        "Game details",
        placeholder="Single-player, Steam Achievements",
        help="Separate multiple details with commas.",
    )

    col3, col4 = st.columns(2)
    with col3:
        achievements = st.number_input(
            "Number of achievements", min_value=0, value=0, step=1
        )
    with col4:
        release_year = st.number_input(
            "Release year", min_value=1970, max_value=2100, value=2024, step=1
        )

    st.caption(
        "Labels are matched case-insensitively to the model's trained "
        "vocabulary. Labels it has never seen are ignored."
    )

    # Keeping your groupmate's important project requirement. 🦆
    like_ducks = st.checkbox("I like ducks 🦆")
    submitted = st.form_submit_button("Predict price")

if submitted:
    if not like_ducks:
        st.warning("Please confirm you like ducks first. 🦆")
    else:
        game = {
            "publisher": publisher,
            "developer": developer,
            "popular_tags": popular_tags,
            "genre": genre,
            "game_details": game_details,
            "achievements": int(achievements),
            "release_year": int(release_year),
        }
        try:
            prediction = predict_price(game)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            st.error(f"Could not make a prediction: {exc}")
        else:
            st.success(f"Estimated Steam price: **${prediction:,.2f}**")
            st.caption("Estimate based on the games in your training data.")
