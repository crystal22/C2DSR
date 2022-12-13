import torch
import torch.nn as nn
from models.GCN import GCN
from models.Attention import SelfAttention


def query_embed(memory, index):
    tmp = list(index.size()) + [-1]
    index = index.view(-1)
    ans = memory(index)
    ans = ans.view(tmp)
    return ans


def query_state(memory, index):
    tmp = list(index.size()) + [-1]
    index = index.view(-1)
    ans = torch.index_select(memory, 0, index)
    ans = ans.view(tmp)
    return ans


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
    def __init__(self, args, adj, adj_single):
        super(C2DSR, self).__init__()
        self.args = args
        self.adj = adj
        self.adj_single = adj_single

        self.d_latent = args.d_latent
        self.n_item = args.n_item
        self.n_item_x = args.n_item_x
        self.n_item_y = args.n_item_y

        self.embed_i = torch.nn.Embedding(self.n_item, self.d_latent, padding_idx=self.n_item - 1)
        if args.shared_item_embed:
            self.embed_i_x = self.embed_i
            self.embed_i_y = self.embed_i
        else:
            self.embed_i_x = torch.nn.Embedding(self.n_item+1, self.d_latent, padding_idx=self.n_item - 1)
            self.embed_i_y = torch.nn.Embedding(self.n_item+2, self.d_latent, padding_idx=self.n_item - 1)

        self.encoder_gnn_x = GCN(args)
        self.encoder_gnn_y = GCN(args)
        self.encoder_gnn = GCN(args)

        self.lin_X = nn.Linear(self.d_latent, self.n_item_x)
        self.lin_Y = nn.Linear(self.d_latent, self.n_item_y)
        self.lin_PAD = nn.Linear(self.d_latent, 1)

        self.encoder_attn = SelfAttention(self.args)
        self.encoder_attn_x = SelfAttention(self.args)
        self.encoder_attn_y = SelfAttention(self.args)

        self.D_X = Discriminator(self.d_latent, self.d_latent)
        self.D_Y = Discriminator(self.d_latent, self.d_latent)

        idx_i = torch.arange(0, self.n_item, 1)
        idx_i_x = torch.arange(0, self.n_item_x, 1)
        idx_i_y = torch.arange(self.n_item_x, self.n_item_x + self.n_item_y, 1)

        self.register_buffer('idx_i', idx_i)
        self.register_buffer('idx_i_x', idx_i_x)
        self.register_buffer('idx_i_y', idx_i_y)

        self.hi_share, self.hi_x, self.hi_y = None, None, None

    def convolve_graph(self):
        feat = query_embed(self.embed_i, self.idx_i)
        feat_x = query_embed(self.embed_i_x, self.idx_i)
        feat_y = query_embed(self.embed_i_y, self.idx_i)

        self.hi_share = self.encoder_gnn(feat, self.adj)
        self.hi_x = self.encoder_gnn_x(feat_x, self.adj_single)
        self.hi_y = self.encoder_gnn_y(feat_y, self.adj_single)

    def forward(self, seq, seq_x, seq_y, pos, pos_x, pos_y):
        seq_gnn_enc = query_state(self.hi_share, seq) + self.embed_i(seq)
        seq_gnn_enc_x = query_state(self.hi_x, seq_x) + self.embed_i_x(seq_x)
        seq_gnn_enc_y = query_state(self.hi_y, seq_y) + self.embed_i_y(seq_y)

        seq_gnn_enc *= self.embed_i.embedding_dim ** 0.5
        seq_gnn_enc_x *= self.embed_i.embedding_dim ** 0.5
        seq_gnn_enc_y *= self.embed_i.embedding_dim ** 0.5

        seq_attn_enc = self.encoder_attn(seq, seq_gnn_enc, pos)
        seq_attn_enc_x = self.encoder_attn_x(seq_x, seq_gnn_enc_x, pos_x)
        seq_attn_enc_y = self.encoder_attn_y(seq_y, seq_gnn_enc_y, pos_y)

        return seq_attn_enc, seq_attn_enc_x, seq_attn_enc_y

    def false_forward(self, seq_neg, pos):
        seq_gnn_enc_neg = query_state(self.hi_share, seq_neg) + self.embed_i(seq_neg)
        seq_gnn_enc_neg *= self.embed_i.embedding_dim ** 0.5
        seq_attn_enc_neg = self.encoder_attn(seq_neg, seq_gnn_enc_neg, pos)

        return seq_attn_enc_neg
