import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


class GCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.n_gnn = args.n_gnn

        self.encoder = []
        for i in range(self.n_gnn):
            self.encoder.append(GCNLayer(args))

        self.encoder = nn.ModuleList(self.encoder)

    def forward(self, fea, adj):
        learn_fea = fea
        tmp_fea = fea
        for layer in self.encoder:
            learn_fea = F.dropout(learn_fea, self.dropout, training=self.training)
            learn_fea = layer(learn_fea, adj)
            tmp_fea = tmp_fea + learn_fea
        return tmp_fea / (self.n_gnn + 1)


class GCNLayer(Module):
    def __init__(self, args, bias=True):
        super().__init__()
        self.in_feat = args.d_latent
        self.out_feat = args.d_latent

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(args.d_latent))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        support = x
        output = torch.spmm(adj, support)

        return output + self.bias if self.bias is not None else output
