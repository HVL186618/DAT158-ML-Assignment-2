from predict import predict_price

game = {
    "popular_tags": ["rpg", "open world", "story rich"],
    "genre": ["action", "rpg"],
    "game_details": ["single-player", "steam achievements"],
    "publisher": ["bethesda softworks"],
    "developer": ["bethesda game studios"],
    "achievements": 50,
    "release_year": 2024
}

price = predict_price(game)

print(f"Predicted price: ${price:.2f}")