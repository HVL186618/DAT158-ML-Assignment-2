import pandas as pd

dataset = "steam_games.csv"
output = "test.csv"
include_free = False
free_threshold = .9
max_price = 90 # suspicious prices start at $96.28

# =======================
# =======================
# =======================

df = pd.read_csv(dataset)

# filter out bundles
df = df[df["types"] == "app"]

def get_price(original, discount):
    if pd.isna(original) or original == "":
        if pd.isna(discount) or discount == "":
            return "error"
        return discount
    return original

def price_number(price):
    try:
        if isinstance(price, str):
            if "free" in price.lower():
                return 0.0

            price = price.replace("$", "").replace(",", "")

        price = float(price)

        return price if 0 <= price <= max_price else "error"

    except (ValueError, TypeError):
        return "error"

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



# select relevant columns


# process silly review data into actual numbers
def unsilly_reviews(review):
    try:
        review = review.split("(")[1]
        count, percent = review.split(")")
        count = count.replace(",", "")

        percent = percent.replace("*", "").strip(",").strip("- ").split("%")[0]
        return count, float(percent)/100.0
    except:
        return "error", "error"
df[["reviews_count", "reviews_positive"]] = df["all_reviews"].apply(
    lambda review: pd.Series(unsilly_reviews(review))
)

df[["recent_count", "recent_positive"]] = df["recent_reviews"].apply(
    lambda review: pd.Series(unsilly_reviews(review))
)

# df = df[df["recent_count"] != "error"] # this filters 38k games down to 2.6k, not worth it
# df = df[df["reviews_count"] != "error"]
for col in [
    "reviews_count", "reviews_positive",
    "recent_count", "recent_positive"
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# filter achievements
def parse_achievements(count):
    try:
        return int(count)
    except ValueError:
        return "error"

df["achievements"] = df["achievements"].apply(
    parse_achievements
)
df = df[df["achievements"] != "error"]

# process release date -> release year
# def release_date(date):
#     try:
#         date = int(date.split("-")[-1])
#         if date > 27: # two dates are before 2000
#             return "error"
#         return date
#     except: # date empty or nan
#         return "error"
# df["release_date"] = df["release_date"].apply(
#     release_date
# )
# df = df[df["release_date"] != "error"]

# release year
dates = df["release_date"].astype("string")

df["release_year"] = pd.to_datetime(
    dates,
    format="mixed",
    dayfirst=True,
    errors="coerce"
).dt.year

# Fix dates like "May-15" and "Jul-16"
short_dates = dates.str.fullmatch(r"[A-Za-z]{3}-\d{2}", na=False)

df.loc[short_dates, "release_year"] = pd.to_datetime(
    dates[short_dates],
    format="%b-%y",
    errors="coerce"
).dt.year

df = df.dropna(subset=["release_year"])
df["release_year"] = df["release_year"].astype(int)

# url and app id might be useful
df["app_id"] = df["url"].str.extract(
    r"/app/(\d+)"
)

df = df.drop_duplicates(subset="app_id")

# developer and publisher + make everything lower case
def lowercase(s):
    if isinstance(s, str):
        return s.lower()
    return s
df["developer"] = df["developer"].apply(lowercase)
df["publisher"] = df["publisher"].apply(lowercase)
df["popular_tags"] = df["popular_tags"].apply(lowercase)
df["game_details"] = df["game_details"].apply(lowercase)

# save
df = df[[
    "name", "recent_count", "recent_positive",
    "reviews_count", "reviews_positive",
    "release_year", "popular_tags", "game_details",
    "achievements", "genre", "price", "url", "app_id",
    "developer", "publisher" # adding these. could make high price estimates better
]]

if not include_free:
    df = df[df["price"] > free_threshold]


# ---
"""
clean suspicious prices
suspicious prices are messing with estimates. especially cheap games being listed under high rices in bundles.
i'll let chatgpt find suspicious listings and check against the actual listing on steam
"""


print(f"Cleaned dataset: {len(df)} games")
print(df.isna().sum())
print(df.dtypes)

df.to_csv(output, index_label="id")

