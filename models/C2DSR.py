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

        self.classifier_x = nn.Linear(self.d_latent, self.n_item_x)
        self.classifier_y = nn.Linear(self.d_latent, self.n_item_y)
        self.classifier_pad = nn.Linear(self.d_latent, 1)
        torch.nn.init.xavier_uniform_(self.classifier_x.weight)
        torch.nn.init.xavier_uniform_(self.classifier_y.weight)
        torch.nn.init.xavier_uniform_(self.classifier_pad.weight)
        nn.init.zeros_(self.classifier_x.bias)
        nn.init.zeros_(self.classifier_y.bias)
        nn.init.zeros_(self.classifier_pad.bias)

        if args.d_bias:
            self.D_x = nn.Bilinear(self.d_latent, self.d_latent, 1, bias=True)
            self.D_y = nn.Bilinear(self.d_latent, self.d_latent, 1, bias=True)
            nn.init.zeros_(self.D_x.bias)
            nn.init.zeros_(self.D_y.bias)
        else:
            self.D_x = nn.Bilinear(self.d_latent, self.d_latent, 1, bias=False)
            self.D_y = nn.Bilinear(self.d_latent, self.d_latent, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.D_x.weight)
        torch.nn.init.xavier_uniform_(self.D_y.weight)

        self.hi_share, self.hi_x, self.hi_y = None, None, None

    def convolve_graph(self):
        self.hi_share = self.gnn_share(self.embed_i.weight, self.adj_share)
        self.hi_x = self.gnn_x(self.embed_i_x.weight, self.adj_specific)
        self.hi_y = self.gnn_y(self.embed_i_y.weight, self.adj_specific)

    def forward(self, seq_share, seq_x, seq_y, pos_share, pos_x, pos_y):
        h_share_gnn = F.embedding(seq_share, self.hi_share) + self.embed_i(seq_share)
        hx_gnn = F.embedding(seq_x, self.hi_x) + self.embed_i_x(seq_x)
        hy_gnn = F.embedding(seq_y, self.hi_y) + self.embed_i_y(seq_y)

        h_share_gnn *= self.d_latent ** 0.5
        hx_gnn *= self.d_latent ** 0.5
        hy_gnn *= self.d_latent ** 0.5

        h_share_attn = self.attn_share(seq_share, h_share_gnn, pos_share)
        hx_attn = self.attn_x(seq_x, hx_gnn, pos_x)
        hy_attn = self.attn_y(seq_y, hy_gnn, pos_y)

        return h_share_attn, hx_attn, hy_attn

    def forward_share(self, seq, pos):
        # forward share embedding layers only
        h_gnn = F.embedding(seq, self.hi_share) + self.embed_i(seq)
        h_gnn *= self.d_latent ** 0.5
        h_attn = self.attn_share(seq, h_gnn, pos)

        return h_attn
