import torch
import pandas as pd
from src.data.lightgcn_data import train_test_split_leave_one_out
from src.train_lightgcn import train_lightgcn


# Load cleaned ratings
ratings = pd.read_parquet("artifacts/ratings_clean.parquet")

# make sure user_id / book_id are zero-indexed
uid_map = {u: i for i, u in enumerate(ratings.user_id.unique())}
iid_map = {i: j for j, i in enumerate(ratings.book_id.unique())}
ratings["u"] = ratings.user_id.map(uid_map)
ratings["i"] = ratings.book_id.map(iid_map)

num_users = ratings.u.nunique()
num_items = ratings.i.nunique()
print(f"Users: {num_users}, Items: {num_items}, Interactions: {len(ratings)}")

train_df, test_df = train_test_split_leave_one_out(ratings)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = train_lightgcn(
    train_df=train_df,
    test_df=test_df,
    num_users=num_users,
    num_items=num_items,
    epochs=10,
    device=device,
    save_path="artifacts/lightgcn.pt",
    batch_size=512,
    eval_sample_users=500,
    eval_every=2
)

books = pd.read_parquet("artifacts/books_clean.parquet")

# Load the model checkpoint
model.load_state_dict(torch.load("artifacts/lightgcn.pt", map_location=device))
model.eval()

# Generate top-N recommendations for a given user
def recommend_for_user(model, user_id, topn=10):
    model.eval()
    with torch.no_grad():
        users_emb, items_emb = model.forward()
        scores = items_emb @ users_emb[user_id]
        top_items = torch.topk(scores, topn).indices.cpu().numpy()
    return top_items

user_id = 0
recommended_items = recommend_for_user(model, user_id, topn=10)

print("\n🎯 Recommended book IDs for user", user_id, ":", recommended_items)


def show_recommendations(recommended_items, books, output_path="artifacts/recommended_books.html"):
    top_books = books.loc[books["book_id"].isin(recommended_items)].copy()
    top_books["image"] = top_books["small_image_url"].map(lambda u: f'<img src="{u}" style="height:80px;border-radius:4px;">')
    html_table = top_books[["image", "title", "authors", "average_rating"]].to_html(
        escape=False, index=False, border=0
    )
    html = f"""
    <html>
    <head><title>Recommended Books</title></head>
    <body style="font-family:Segoe UI, sans-serif;margin:30px;background:#fafafa;">
      <h2>Recommended Books for User {user_id}</h2>
      {html_table}
    </body>
    </html>
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Recommendations saved to {output_path}")

show_recommendations(recommended_items, books)

# --- FINAL EVALUATION & ARTIFACTS ---
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.eval import precision_recall_ndcg  # adjust to your import style if needed

# use the saved metrics list that you already populated in training
# assume metrics list: metrics = [(epoch, prec, rec, ndcg), ...]
df_metrics = pd.DataFrame(metrics, columns=["epoch","precision","recall","ndcg"])
df_metrics.to_csv("artifacts/training_metrics.csv", index=False)

# Plot (if not already plotted)
plt.figure(figsize=(8,5))
plt.plot(df_metrics['epoch'], df_metrics['precision'], marker='o', label='Precision@10')
plt.plot(df_metrics['epoch'], df_metrics['recall'], marker='o', label='Recall@10')
plt.plot(df_metrics['epoch'], df_metrics['ndcg'], marker='o', label='NDCG@10')
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Training metrics per epoch")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("artifacts/training_metrics.png", dpi=300)
print("Saved artifacts/training_metrics.png and training_metrics.csv")

# -----------------------
# Final evaluation on test set (full or sampled)
# -----------------------
# If you can do full-eval (all users), set sample_users=None in the function.
# Otherwise set sample_users=2000 or 5000 for a representative sample.
Ks = [5, 10, 20]  # report multiple Ks in paper
final_results = []
for K in Ks:
    prec, rec, ndcg = precision_recall_ndcg(
        model,
        train_df,
        test_df,
        num_users,
        num_items,
        Ks=[K],
        sample_users=None,   # set None to evaluate all users — be careful with runtime
        device=device
    )
    final_results.append({"K": K, "precision": prec, "recall": rec, "ndcg": ndcg})

df_final = pd.DataFrame(final_results)
df_final.to_csv("artifacts/final_eval.csv", index=False)
print("Saved final evaluation to artifacts/final_eval.csv")
print(df_final.to_string(index=False))

# --- FINAL EVALUATION & ARTIFACTS ---
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.eval import precision_recall_ndcg  # adjust to your import style if needed

# use the saved metrics list that you already populated in training
# assume metrics list: metrics = [(epoch, prec, rec, ndcg), ...]
df_metrics = pd.DataFrame(metrics, columns=["epoch","precision","recall","ndcg"])
df_metrics.to_csv("artifacts/training_metrics.csv", index=False)

# Plot (if not already plotted)
plt.figure(figsize=(8,5))
plt.plot(df_metrics['epoch'], df_metrics['precision'], marker='o', label='Precision@10')
plt.plot(df_metrics['epoch'], df_metrics['recall'], marker='o', label='Recall@10')
plt.plot(df_metrics['epoch'], df_metrics['ndcg'], marker='o', label='NDCG@10')
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Training metrics per epoch")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("artifacts/training_metrics.png", dpi=300)
print("Saved artifacts/training_metrics.png and training_metrics.csv")



