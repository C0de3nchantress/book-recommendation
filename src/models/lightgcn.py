# src/models/lightgcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_size=64, n_layers=3, adj=None, device='cpu'):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.device = device

        self.embedding = nn.Embedding(self.n_nodes, emb_size)
        nn.init.xavier_uniform_(self.embedding.weight)

        # adjacency should be precomputed normalized sparse matrix (scipy) -> torch sparse tensor
        if adj is not None:
            self.adj = self._convert_sp_mat_to_torch_sparse(adj).to(device)
        else:
            self.adj = None

    def _convert_sp_mat_to_torch_sparse(self, mat):
        mat_coo = mat.tocoo().astype('float32')
        indices = torch.from_numpy(np.vstack((mat_coo.row, mat_coo.col)).astype(np.int64))
        # ensure values are float32 tensor
        values = torch.from_numpy(mat_coo.data.astype(np.float32))
        shape = mat_coo.shape
        # use sparse_coo_tensor (recommended) and coalesce to ensure indices are unique/sorted
        sp_t = torch.sparse_coo_tensor(indices, values, torch.Size(shape), dtype=torch.float32, device=self.device)
        return sp_t.coalesce()

    def forward(self):
        """
        Propagate embeddings and return final user and item embeddings.
        """
        all_emb = self.embedding.weight  # (n_nodes, emb)
        embs = [all_emb]

        # propagate
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)  # (n_nodes, n_layers+1, emb)
        final_emb = torch.mean(embs, dim=1)  # mean of layers
        users, items = final_emb[:self.n_users], final_emb[self.n_users:]
        return users, items

    def get_user_item_emb(self, users, items):
        u_embs, i_embs = self.forward()
        return u_embs[users], i_embs[items]
