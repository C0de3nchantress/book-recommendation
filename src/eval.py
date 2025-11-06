# src/eval.py
import numpy as np
import torch
from scipy.sparse import csr_matrix
from typing import List


def precision_recall_ndcg(model, train_df, test_df, num_users, num_items, Ks=[10], sample_users=1000, device='cpu', batch_size=128):
    """Vectorized evaluation: score users in batches to avoid Python-per-user scoring overhead.

    Args:
        model: LightGCN model with forward() returning user and item embeddings.
        train_df/test_df: DataFrames with columns 'u' and 'i'.
        sample_users: number of users to sample for evaluation (keeps eval fast).
        batch_size: how many users to score at once.
    """
    # build train/test interactions dicts
    train_user_items = train_df.groupby('u')['i'].apply(set).to_dict()
    test_user_items = test_df.groupby('u')['i'].apply(set).to_dict()
    all_users = np.array(list(train_user_items.keys()))
    if len(all_users) > sample_users:
        users = np.random.choice(all_users, sample_users, replace=False)
    else:
        users = all_users

    precs, recs, ndcgs = [], [], []
    model.eval()
    with torch.no_grad():
        users_emb, items_emb = model.forward()
        users_emb = users_emb.cpu().numpy()
        items_emb = items_emb.cpu().numpy()

        K = Ks[0]
        discounts = 1.0 / np.log2(np.arange(2, K + 2))

        # process users in batches
        for start in range(0, len(users), batch_size):
            batch_users = users[start:start + batch_size]
            batch_u_emb = users_emb[batch_users]  # (b, d)
            # scores: (b, n_items)
            scores = batch_u_emb.dot(items_emb.T)

            # filter seen items for each user in batch (loop over batch only)
            for r_idx, u in enumerate(batch_users):
                train_pos = train_user_items.get(int(u), set())
                if train_pos:
                    scores[r_idx, list(train_pos)] = -np.inf

            # get top-K per row
            # argpartition to get K unsorted indices per row
            topk_idx = np.argpartition(-scores, K, axis=1)[:, :K]  # (b, K)
            # sort the K candidates per row
            row_idx = np.arange(topk_idx.shape[0])[:, None]
            topk_sorted = topk_idx[row_idx, np.argsort(-scores[row_idx, topk_idx], axis=1)]

            # compute metrics per user in batch
            for r_idx, u in enumerate(batch_users):
                test_pos = test_user_items.get(int(u), set())
                if len(test_pos) == 0:
                    continue
                topk = topk_sorted[r_idx]
                hits = np.isin(topk, list(test_pos)).astype(int)
                prec = hits.sum() / K
                rec = hits.sum() / len(test_pos)
                dcg = (hits * discounts).sum()
                idcg = (1.0 / np.log2(np.arange(2, min(len(test_pos), K) + 2))).sum()
                ndcg = dcg / idcg if idcg > 0 else 0.0
                precs.append(prec)
                recs.append(rec)
                ndcgs.append(ndcg)

    # handle case with no users evaluated
    if len(precs) == 0:
        return 0.0, 0.0, 0.0
    return np.mean(precs), np.mean(recs), np.mean(ndcgs)
