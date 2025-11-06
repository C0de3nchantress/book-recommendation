# src/data/lightgcn_data.py
import numpy as np
import scipy.sparse as sp
import pandas as pd
from typing import Tuple, Dict

def build_interaction_matrix(train_df: pd.DataFrame, num_users: int, num_items: int) -> sp.csr_matrix:
    """
    train_df must have 'u' and 'i' integer columns (0-indexed).
    Returns item-user sparse matrix (users rows, items cols).
    """
    rows = train_df['u'].to_numpy()
    cols = train_df['i'].to_numpy()
    data = np.ones(len(train_df), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(num_users, num_items))

def normalize_adj(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Symmetric normalization D^-1/2 * A * D^-1/2 (for adjacency with self-loops if desired)."""
    rowsum = np.array(adj.sum(axis=1)).squeeze()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum>0)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt.dot(adj).dot(D_inv_sqrt)

def create_adj_matrix(interaction: sp.csr_matrix) -> sp.csr_matrix:
    """Build bipartite adjacency ( (U+I) x (U+I) )."""
    num_users, num_items = interaction.shape
    # top-right is interaction, bottom-left is interaction.T
    upper = sp.hstack([sp.csr_matrix((num_users, num_users)), interaction])
    lower = sp.hstack([interaction.T, sp.csr_matrix((num_items, num_items))])
    adj = sp.vstack([upper, lower]).tocsr()
    return adj

def train_test_split_leave_one_out(ratings: pd.DataFrame, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user leave-one-out: keep one random interaction per user for test."""
    rng = np.random.default_rng(seed)
    # For each user group, sample one row and collect its index (scalar) so we have a flat list
    sampled_idx = ratings.groupby('u').apply(lambda g: g.sample(n=1, random_state=seed).index[0]).values
    # sampled_idx is an array of index labels (e.g., integers) safe to use with .loc and .drop
    test = ratings.loc[sampled_idx]
    train = ratings.drop(index=sampled_idx)
    return train.reset_index(drop=True), test.reset_index(drop=True)
