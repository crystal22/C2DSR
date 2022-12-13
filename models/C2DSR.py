import torch
import torch.nn as nn
import torch.nn.functional as F

from models.GCN import GCN
from models.Attention import SelfAttention


class Discriminator(torch.nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()

        if bias:
            self.f_k = nn.Bilinear(d_in, d_out, 1, bias=True)
            nn.init.zeros_(self.f_k.bias)
        else:
            self.f_k = nn.Bilinear(d_in, d_out, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.f_k.weight.data)

    def forward(self, S, node):
        return self.f_k(node, S)


class C2DSR(torch.nn.Module):
    def __init__(self, args, adj, adj_specific):
        super(C2DSR, self).__init__()
        self.args = args
        self.adj_share = adj
        self.adj_specific = adj_specific

        self.d_latent = args.d_latent
        self.n_item = args.n_item
        self.n_item_x = args.n_item_x
        self.n_item_y = args.n_item_y

        self.embed_i = torch.nn.Embedding(self.n_item, self.d_latent, padding_idx=self.n_item - 1)
        if args.shared_item_embed:
            self.embed_i_x = self.embed_i
            self.embed_i_y = self.embed_i
        else:
            self.embed_i_x = torch.nn.Embedding(self.n_item, self.d_latent, padding_idx=self.n_item - 1)
            self.embed_i_y = torch.nn.Embedding(self.n_item, self.d_latent, padding_idx=self.n_item - 1)

        self.gnn_share = GCN(args)
        self.gnn_x = GCN(args)
        self.gnn_y = GCN(args)

        self.attn_share = SelfAttention(self.args)
        self.attn_x = SelfAttention(self.args)
        self.attn_y = SelfAttention(self.args)

        self.lin_X = nn.Linear(self.d_latent, self.n_item_x)
        self.lin_Y = nn.Linear(self.d_latent, self.n_item_y)
        self.lin_PAD = nn.Linear(self.d_latent, 1)

        self.D_X = Discriminator(self.d_latent, self.d_latent)
        self.D_Y = Discriminator(self.d_latent, self.d_latent)

        self.hi_share, self.hi_x, self.hi_y = None, None, None

    def convolve_graph(self):
        self.hi_share = self.gnn_share(self.embed_i.weight, self.adj_share)
        self.hi_x = self.gnn_x(self.embed_i_x.weight, self.adj_specific)
        self.hi_y = self.gnn_y(self.embed_i_y.weight, self.adj_specific)

        # self.hi_share = self.embed_i.weight
        # self.hi_x = self.embed_i_x.weight
        # self.hi_y = self.embed_i_y.weight

    def forward(self, seq, seq_x, seq_y, pos, pos_x, pos_y):
        seq_gnn_enc = F.embedding(seq, self.hi_share) + self.embed_i(seq)
        seq_gnn_enc_x = F.embedding(seq_x, self.hi_x) + self.embed_i_x(seq_x)
        seq_gnn_enc_y = F.embedding(seq_y, self.hi_y) + self.embed_i_y(seq_y)

        seq_gnn_enc *= self.d_latent ** 0.5
        seq_gnn_enc_x *= self.d_latent ** 0.5
        seq_gnn_enc_y *= self.d_latent ** 0.5

        seq_attn_enc = self.attn_share(seq, seq_gnn_enc, pos)
        seq_attn_enc_x = self.attn_x(seq_x, seq_gnn_enc_x, pos_x)
        seq_attn_enc_y = self.attn_y(seq_y, seq_gnn_enc_y, pos_y)

        return seq_attn_enc, seq_attn_enc_x, seq_attn_enc_y

    def forward_negative(self, seq_neg, pos):
        seq_gnn_enc_neg = F.embedding(seq_neg, self.hi_share) + self.embed_i(seq_neg)
        seq_gnn_enc_neg *= self.d_latent ** 0.5
        seq_attn_enc_neg = self.attn_share(seq_neg, seq_gnn_enc_neg, pos)

        return seq_attn_enc_neg
