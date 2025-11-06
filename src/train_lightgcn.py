# src/train_lightgcn.py
import torch
import numpy as np
import scipy.sparse as sp
from tqdm import trange
import pandas as pd
from scipy.sparse import csr_matrix
from torch.optim import Adam

from .data.lightgcn_data import build_interaction_matrix, create_adj_matrix, normalize_adj, train_test_split_leave_one_out
from .models.lightgcn import LightGCN
from .eval import precision_recall_ndcg

def sample_mini_batch(train_mat: csr_matrix, batch_size):
    """Uniformly sample positive (u,i) pairs and random negative j not in user's positives."""
    num_users = train_mat.shape[0]
    users = np.random.randint(0, num_users, size=batch_size)
    pos_items = []
    neg_items = []
    for u in users:
        row = train_mat.getrow(u).indices
        if len(row) == 0:
            pos = np.random.randint(0, train_mat.shape[1])
        else:
            pos = np.random.choice(row)
        # negative
        neg = np.random.randint(0, train_mat.shape[1])
        while neg in row:
            neg = np.random.randint(0, train_mat.shape[1])
        pos_items.append(pos)
        neg_items.append(neg)
    return users, np.array(pos_items), np.array(neg_items)

def bpr_loss(u_emb, pos_emb, neg_emb, reg=1e-4):
    pos_scores = (u_emb * pos_emb).sum(dim=1)
    neg_scores = (u_emb * neg_emb).sum(dim=1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).sum()
    # l2 reg
    loss += reg * (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2))
    return loss

def train_lightgcn(train_df: pd.DataFrame, test_df: pd.DataFrame, num_users:int, num_items:int, epochs=50, device='cpu', save_path: str = None, eval_every: int = 5, batch_size: int = 1024, eval_sample_users: int = 1000):
    train_mat = build_interaction_matrix(train_df, num_users, num_items)
    adj = create_adj_matrix(train_mat)
    norm_adj = normalize_adj(adj + sp.eye(adj.shape[0]))  # + self loops optional
    model = LightGCN(num_users, num_items, emb_size=64, n_layers=3, adj=norm_adj, device=device).to(device)
    opt = Adam(model.parameters(), lr=1e-3)

    train_csr = train_mat.tocsr()
    best_ndcg = -1.0
    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        iters = max(1, (train_df.shape[0] // batch_size) + 1)
        for _ in range(iters):
            users, pos, neg = sample_mini_batch(train_csr, batch_size=batch_size)
            users_t = torch.LongTensor(users).to(device)
            pos_t = torch.LongTensor(pos).to(device)
            neg_t = torch.LongTensor(neg).to(device)

            # compute full propagated embeddings once per batch (heavy op) and index
            u_all, i_all = model.forward()
            # ensure embeddings are on the correct device
            if u_all.device != users_t.device:
                u_all = u_all.to(device)
                i_all = i_all.to(device)

            u_emb = u_all[users_t]
            i_pos_emb = i_all[pos_t]
            i_neg_emb = i_all[neg_t]

            loss = bpr_loss(u_emb, i_pos_emb, i_neg_emb, reg=1e-4)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        # eval every few epochs
        # evaluation
        if epoch % eval_every == 0 or epoch==1:
            print(f"Epoch {epoch} avg loss {np.mean(losses):.4f}")
            # run light eval on small sample for speed
            prec, rec, ndcg = precision_recall_ndcg(model, train_df, test_df, num_users, num_items, Ks=[10], sample_users=eval_sample_users, device=device)
            print(f"Precision@10 {prec:.4f} Recall@10 {rec:.4f} NDCG@10 {ndcg:.4f}")
            # checkpoint best model by NDCG
            if save_path is not None:
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    print(f"New best NDCG {best_ndcg:.4f} - saving model to {save_path}")
                    torch.save(model.state_dict(), save_path)
    return model
