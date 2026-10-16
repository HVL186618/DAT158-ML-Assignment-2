import pandas as pd

df = pd.read_csv("steam_games.csv")

# filter out bundles
df = df[df["types"] == "app"]

def get_price(original, discount):
    if pd.isna(original) or original == "":
        if pd.isna(discount) or discount == "":
            return "error"
        return discount
    return original

def price_number(price):
    if isinstance(price, str):
        if "$" in price:
            return float(price[1:])
        elif "free" in price.lower():
            return 0
        try:
            return float(price)
        except:
            return "error"
    return price

# process price
df["price"] = df.apply(
    lambda row: get_price(
        row["original_price"],
        row["discount_price"]
    ),
    axis=1
)
df["price"] = df["price"].apply(
    price_number
)

df = df[df["price"] != "error"]
#print(df["price"])

print()


# select relevant columns


# process silly review data into actual numbers
def unsilly_reviews(review):
    try:
        review = review.split("(")[1]
        count, percent = review.split(")")
        count = count.replace(",", "")

        percent = percent.strip(",- ").split("%")[0]
        return count, float(percent)/100.0
    except:
        return "error", "error"
df[["reviews_count", "reviews_positive"]] = df["all_reviews"].apply(
    lambda review: pd.Series(unsilly_reviews(review))
)

df[["recent_count", "recent_positive"]] = df["recent_reviews"].apply(
    lambda review: pd.Series(unsilly_reviews(review))
)

# process release date -> release year

# save
df = df[[
    "name", "recent_count", "recent_positive",
    "reviews_count", "reviews_positive",
    "release_date", "popular_tags", "game_details",
    "achievements", "genre", "price"
]]
df.to_csv("test.csv")