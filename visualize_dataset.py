import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("test.csv")

# price histogram
df["price"].hist(bins=50)
df[df["price"] > 0]["price"].hist(bins=50)

plt.xlabel("Price ($)")
plt.ylabel("Number of games")
plt.title("Steam Game Price Distribution")
plt.xlim(0, 100)

plt.show()

# price by year
price_by_year = df.groupby("release_year")["price"].mean()

plt.plot(
    price_by_year.index,
    price_by_year.values,
    marker="o"
)

plt.xlabel("Release Year")
plt.ylabel("Median Price ($)")
plt.title("Mean Steam Game Price by Release Year")
plt.grid(True)

plt.show()

# ----------------------------
# genres
genres = df[["genre", "price"]].copy()

# turn comma-separated genres into lists
genres["genre"] = genres["genre"].str.split(",")

# give each genre its own row
genres = genres.explode("genre")

# calculate average price for each genre
average_prices = (
    genres.groupby("genre")["price"]
    .mean()
    .sort_values()
)
average_prices.plot(kind="barh", figsize=(10, 7))

plt.xlabel("Average Price ($)")
plt.ylabel("Genre")
plt.title("Average Steam Game Price by Genre")
plt.tight_layout()
genre_stats = genres.groupby("genre")["price"].agg(
    ["mean", "count"]
)
print(genre_stats.sort_values("mean", ascending=False))
plt.show()

# ----------
# tags
tags = df[["popular_tags", "price"]].copy()
tags["popular_tags"] = tags["popular_tags"].str.split(",")
tags = tags.explode("popular_tags")
tags["popular_tags"] = tags["popular_tags"].str.strip()

tag_stats = (
    tags.groupby("popular_tags")["price"]
    .agg(["mean", "count"])
)

# gnore tags with fewer than 100 games
tag_stats = tag_stats[tag_stats["count"] >= 100]

# select highest-priced tags
average_prices = (
    tag_stats.nlargest(100, "mean")["mean"]
    .sort_values()
)

average_prices.plot(kind="barh", figsize=(10, 8))

plt.xlabel("Average Price ($)")
plt.ylabel("Tag")
plt.title("Steam Game Prices by Tag")
plt.tight_layout()
plt.show()