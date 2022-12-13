import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout_gnn = args.dropout_gnn
        self.n_gnn = args.n_gnn

    def forward(self, h, adj):
        h_sum = [h]
        for _ in range(self.n_gnn):
            h = F.dropout(h, self.dropout_gnn, training=self.training)
            h = torch.spmm(adj, h)
            h_sum.append(h)
        return torch.stack(h_sum, dim=1).mean(dim=1)
