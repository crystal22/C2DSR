import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders import SelfAttention, GCN


class C2DSR(torch.nn.Module):
    def __init__(self, args, adj, adj_specific):
        super(C2DSR, self).__init__()
        self.args = args
        self.adj_share = adj
        self.adj_specific = adj_specific

        self.d_latent = args.d_latent
        self.n_item = args.n_item
        self.n_item_a = args.n_item_a
        self.n_item_b = args.n_item_b

        self.embed_i = torch.nn.Embedding(self.n_item, self.d_latent, padding_idx=self.n_item - 1)
        if args.shared_item_embed:
            self.embed_i_a = self.embed_i
            self.embed_i_b = self.embed_i
        else:
            self.embed_i_a = torch.nn.Embedding(self.n_item, self.d_latent, padding_idx=self.n_item - 1)
            self.embed_i_b = torch.nn.Embedding(self.n_item, self.d_latent, padding_idx=self.n_item - 1)

        self.gnn_share = GCN(args)
        self.gnn_a = GCN(args)
        self.gnn_b = GCN(args)

        self.attn_share = SelfAttention(self.args)
        self.attn_a = SelfAttention(self.args)
        self.attn_b = SelfAttention(self.args)

        self.classifier_a = nn.Linear(self.d_latent, self.n_item_a)
        self.classifier_b = nn.Linear(self.d_latent, self.n_item_b)
        self.classifier_pad = nn.Linear(self.d_latent, 1)
        torch.nn.init.xavier_uniform_(self.classifier_a.weight)
        torch.nn.init.xavier_uniform_(self.classifier_b.weight)
        torch.nn.init.xavier_uniform_(self.classifier_pad.weight)
        nn.init.zeros_(self.classifier_a.bias)
        nn.init.zeros_(self.classifier_b.bias)
        nn.init.zeros_(self.classifier_pad.bias)

        if args.d_bias:
            self.D_a = nn.Bilinear(self.d_latent, self.d_latent, 1, bias=True)
            self.D_b = nn.Bilinear(self.d_latent, self.d_latent, 1, bias=True)
            nn.init.zeros_(self.D_a.bias)
            nn.init.zeros_(self.D_b.bias)
        else:
            self.D_a = nn.Bilinear(self.d_latent, self.d_latent, 1, bias=False)
            self.D_b = nn.Bilinear(self.d_latent, self.d_latent, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.D_a.weight)
        torch.nn.init.xavier_uniform_(self.D_b.weight)

        self.hi_share, self.hi_a, self.hi_b = None, None, None

    def convolve_graph(self):
        self.hi_share = self.gnn_share(self.embed_i.weight, self.adj_share)
        self.hi_a = self.gnn_a(self.embed_i_a.weight, self.adj_specific)
        self.hi_b = self.gnn_b(self.embed_i_b.weight, self.adj_specific)

    def forward(self, seq_share, seq_a, seq_b, pos_share, pos_a, pos_b):
        h_share_gnn = F.embedding(seq_share, self.hi_share) + self.embed_i(seq_share)
        hx_gnn = F.embedding(seq_a, self.hi_a) + self.embed_i_a(seq_a)
        hy_gnn = F.embedding(seq_b, self.hi_b) + self.embed_i_b(seq_b)

        h_share_gnn *= self.d_latent ** 0.5
        hx_gnn *= self.d_latent ** 0.5
        hy_gnn *= self.d_latent ** 0.5

        h_share_attn = self.attn_share(seq_share, h_share_gnn, pos_share)
        hx_attn = self.attn_a(seq_a, hx_gnn, pos_a)
        hy_attn = self.attn_b(seq_b, hy_gnn, pos_b)

        return h_share_attn, hx_attn, hy_attn

    def forward_share(self, seq, pos):
        # forward share embedding layers only
        h_gnn = F.embedding(seq, self.hi_share) + self.embed_i(seq)
        h_gnn *= self.d_latent ** 0.5
        h_attn = self.attn_share(seq, h_gnn, pos)

        return h_attn
