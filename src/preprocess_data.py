import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.ticker import PercentFormatter
import matplotlib
from scipy.stats import pearsonr


ratings = pd.read_csv('data/ratings.csv') #this has user_id, book_id, rating
books = pd.read_csv("data/books.csv") # this has book metadata

ratings['N'] = ratings.groupby(['user_id', 'book_id'])['book_id'].transform('count') 

# duplicate check
num_duplicates = len(ratings[ratings["N"] > 1])
print(f"Number of duplicate ratings: {num_duplicates}")

#duplicate fix
ratings = ratings[ratings["N"] == 1].copy()

# removing users with less than 5 reviews as they're basically noise

ratings["N"] = ratings.groupby("user_id")["book_id"].transform("count")
num_users_with_ratings_less_than_5 = ratings.loc[ratings["N"] < 5, "user_id"].nunique()

print(f"number of users who rated fewer than 5 books: {num_users_with_ratings_less_than_5}")
ratings = ratings[ratings["N"] > 4].copy()

# selecting only a subset of users to lower calculation time

np.random.seed(5)
user_fraction = 1

unique_users = ratings["user_id"].unique()
sample_users = np.random.choice(unique_users, size=round(user_fraction * len(unique_users)), replace=False)

print(f"Number of ratings (before): {len(ratings)}")

ratings = ratings[ratings["user_id"].isin(sample_users)].copy()

print(f"Number of ratings (after): {len(ratings)}")

# How are the ratings distributed
plt.figure(figsize=(6,4))
sns.countplot(data=ratings, x="rating", hue="rating", palette="YlGnBu", legend=False, edgecolor="grey")
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("artifacts/rating_distribution.png", dpi=300, bbox_inches="tight")
print("Plot saved to artifacts/rating_distribution.png")

# Number of ratings per user
user_counts = ratings.groupby("user_id")["rating"].count().reset_index(name="number_of_ratings_per_user")

plt.figure(figsize=(7,4))
sns.histplot(
    data=user_counts,
    x="number_of_ratings_per_user",
    color="green",
    edgecolor="black",
)

plt.title("Number of Ratings per User (Zoomed in 3-50)")
plt.xlabel("Ratings per User")
plt.ylabel("Count of Users")
plt.xlim(3, 50)

plt.tight_layout()
plt.savefig("artifacts/ratings_per_user.png", dpi=300, bbox_inches="tight")
print("Plot saved to artifacts/ratings_per_user.png")

# Mean user ratings
mean_user_ratings = ratings.groupby("user_id")["rating"].mean().reset_index(name="mean_user_rating")

plt.figure(figsize=(7,4))
plt.hist(
    mean_user_ratings["mean_user_rating"],
    bins=30,
    color="cadetblue",
    edgecolor="grey"
)
plt.title("Distribution of Mean User Ratings")
plt.xlabel("Mean User Rating")
plt.ylabel("Count of Users")
plt.tight_layout()
plt.savefig("artifacts/mean_user_rating_distribution.png", dpi=300, bbox_inches="tight")
print("Plot saved to artifacts/mean_user_rating_distribution.png")

# Number of ratings per book
book_counts = ratings.groupby("book_id")["rating"].count().reset_index(name="number_of_ratings_per_book")

plt.figure(figsize=(7,4))
plt.hist(
    book_counts["number_of_ratings_per_book"],
    bins=40,
    color="orange",
    edgecolor="grey",
    range=(0, 40)
)
plt.title("Number of Ratings per Book")
plt.xlabel("Number of Ratings per Book")
plt.ylabel("Count of Books")
plt.xlim(0, 40)
plt.tight_layout()
plt.savefig("artifacts/ratings_per_book.png", dpi=300, bbox_inches="tight")
print("Plot saved to artifacts/ratings_per_book.png")

# Distribution of mean book ratings
mean_book_ratings = ratings.groupby("book_id")["rating"].mean().reset_index(name="mean_book_rating")

plt.figure(figsize=(7,4))
plt.hist(
    mean_book_ratings["mean_book_rating"],
    bins=30,
    color="orange",
    edgecolor="grey",
    range=(1, 5)
)
plt.title("Distribution of Mean Book Ratings")
plt.xlabel("Mean Book Rating")
plt.ylabel("Count of Books")
plt.xlim(1, 5)
plt.tight_layout()
plt.savefig("artifacts/mean_book_rating_distribution.png", dpi=300, bbox_inches="tight")
print("Plot saved to artifacts/mean_book_rating_distribution.png")

# Genre analysis

book_tags = pd.read_csv("data/book_tags.csv")
tags = pd.read_csv("data/tags.csv")

genres = [
    "art", "biography", "business", "chick lit", "children's", "christian", "classics", 
    "comics", "contemporary", "cookbooks", "crime", "ebooks", "fantasy", "fiction","graphic novels", "historical fiction", "history", "horror", 
    "humor and comedy", "manga", "memoir", "music", "mystery", "nonfiction", 
    "paranormal", "philosophy", "poetry", "psychology", "religion", "romance", 
    "science", "science fiction", "self help", "suspense", "spirituality", 
    "sports", "thriller", "travel", "young adult"
]

exclude_genres = ["fiction", "nonfiction", "ebooks", "contemporary"] # cuz they basic
genres = [g for g in genres if g not in exclude_genres]

# Finding available genres

tags['tag_name_lower'] = tags['tag_name'].str.lower()
available_genres = [g for g in genres if g in tags['tag_name_lower'].values]
available_tags = tags[tags['tag_name_lower'].isin(available_genres)]['tag_id'].tolist()

# calculating genre statistics

tmp = (book_tags[book_tags['tag_id'].isin(available_tags)]
       .groupby('tag_id')
       .size()
       .reset_index(name='n'))

tmp['sumN'] = tmp['n'].sum()
tmp['percentage'] = tmp['n'] / tmp['sumN']
tmp = tmp.sort_values('percentage', ascending=False)

# getting tag names by joining with tags
tmp = tmp.merge(tags[['tag_id', 'tag_name']], on='tag_id', how='left')

fig, ax = plt.subplots(figsize=(10, 8))

# Normalize 0–1 since tmp['percentage'] is a fraction
norm = colors.Normalize(vmin=0, vmax=tmp['percentage'].max())
cmap = matplotlib.colormaps.get_cmap('YlOrRd')

# Color each bar by its percentage
colors_for_bars = cmap(norm(tmp['percentage'].values))
bars = ax.barh(range(len(tmp)), tmp['percentage'], color=colors_for_bars)

# Axis labels & ticks
ax.set_yticks(range(len(tmp)))
ax.set_yticklabels(tmp['tag_name'])
ax.set_xlabel('Percentage')
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_ylabel('Genre')
ax.set_title('Genre Distribution in Book Tags')
ax.invert_yaxis()

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # required in some versions
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Percentage')
cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))

fig.tight_layout()
fig.savefig("artifacts/genre_distribution.png", dpi=300, bbox_inches="tight")
print("Plot saved to artifacts/genre_distribution.png")


# top 10 rated books

# sorting top 10 by rating
top_rated_books = (
    books
    .assign(image=lambda d: d['small_image_url'].map(lambda u: f'<img src="{u}" style="height:80px;border-radius:4px;">'))
    .sort_values('average_rating', ascending=False)
    .nlargest(10, 'average_rating')
    [['image', 'title', 'ratings_count', 'average_rating']]
)

def html_basic_page(html_table):
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Top Rated Books</title>
<style>
  body {{
    font-family: "Segoe UI", sans-serif;
    margin: 30px;
    background: #fafafa;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    background: #fff;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
  }}
  th, td {{
    padding: 10px 14px;
    border-bottom: 1px solid #e5e5e5;
    text-align: left;
    vertical-align: middle;
  }}
  th {{
    background: #f2f2f2;
    font-weight: 600;
  }}
  tr:hover {{
    background-color: #f9f9f9;
  }}
  img {{
    height: 70px;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }}
</style>
</head>
<body>
<h2>Top Rated Books</h2>
{html_table}
</body>
</html>
"""

# making an html file
html_table = top_rated_books.to_html(
    escape=False, 
    index=False, 
    classes='nowrap hover row-border', 
    border=0
)

html = html_basic_page(html_table)

output_path = "artifacts/top_books_by_ratings.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"HTML file saved at: {output_path}")


# Top 10 popular books

top_popular_books = (
    books
    .assign(image=lambda d: d['small_image_url'].map(lambda u: f'<img src="{u}" style="height:80px;border-radius:4px;">'))
    .sort_values('ratings_count', ascending=False)
    .nlargest(10, 'ratings_count')
    [['image', 'title', 'ratings_count', 'average_rating']]
)

html_table = top_popular_books.to_html(
    escape=False, 
    index=False, 
    classes='nowrap hover row-border', 
    border=0
)

html = html_basic_page(html_table)

output_path = "artifacts/top_books_by_popularity.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"HTML table saved successfully at: {output_path}")


# What influences a book's rating (correlation matrix)

cols = [
    "books_count",
    "original_publication_year",
    "ratings_count",
    "work_ratings_count",
    "work_text_reviews_count",
    "average_rating"
]
tmp = books[cols].copy()

#correlation matrix wuhuuuu
corr = tmp.corr(method='pearson', numeric_only=True)

# lower triangle correlation heatmap wuhhuuuu
mask = ~np.tril(np.ones_like(corr, dtype=bool)) 

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr, 
    mask=mask,
    cmap="YlOrRd", 
    annot=True, 
    fmt=".2f", 
    cbar_kws={"label": "Correlation"},
    square=True,
    linewidths=0.5
)
plt.title("Correlation Between Book Features", fontsize=14, pad=12)
plt.tight_layout()

plt.savefig("artifacts/book_correlation.png", dpi=300, bbox_inches="tight")
print("Correlation heatmap saved to artifacts/book_correlation.png")

# Doesnt seem to be correlated at all so people rate books based on quality and aren't influenced by other factors

# Relationship between number of ratings and average rating

books_filtered = books[books["ratings_count"] < 1e5]

# Computing correlation coefficient
r_value, _ = pearsonr(books_filtered["ratings_count"], books_filtered["average_rating"])
r_label = f"$r = {r_value:.2f}$" 

fig, ax = plt.subplots(figsize=(9, 7))

#density heatmap

hb = ax.hexbin(
    books_filtered["ratings_count"],
    books_filtered["average_rating"],
    gridsize=50,
    cmap="Spectral"
)

#regression line
sns.regplot(
    data=books_filtered,
    x="ratings_count",
    y="average_rating",
    scatter=False,
    color="orchid",
    line_kws={"linewidth": 2},
    ax=ax
)

cbar = fig.colorbar(hb, ax=ax)
cbar.set_label("Count (hex density)")

ax.text(
    x=85000, 
    y=2.7, 
    s=r_label,
    fontsize=14,
    color="orchid",
    fontstyle="italic"
)


ax.set_xlabel("Ratings Count")
ax.set_ylabel("Average Rating")
ax.set_title("Relationship Between Popularity and Average Rating")
sns.despine()

plt.tight_layout()

plt.savefig("artifacts/popularity_vs_rating.png", dpi=300, bbox_inches="tight")
print("Plot saved to artifacts/popularity_vs_rating.png")


ratings[['user_id', 'book_id', 'rating']].to_parquet("artifacts/ratings_clean.parquet")
books.to_parquet("artifacts/books_clean.parquet")
print("Saved cleaned datasets for model training!")
