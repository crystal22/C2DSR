import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch.nn.functional as F


class SelfAttention(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.idx_pad = args.idx_pad

        # attn_mask = torch.tril(torch.ones((args.len_max, args.len_max)) == 1)
        # attn_mask = attn_mask.float().masked_fill(attn_mask == 0, -torch.inf).masked_fill(attn_mask == 1, 0)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(args.len_max)
        self.register_buffer('attn_mask', attn_mask)

        self.dropout_attn = nn.Dropout(p=args.dropout_attn)
        self.pos_emb = nn.Embedding(args.len_max, args.d_latent)  # TO IMPROVE

        # loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.encoder_layer = TransformerEncoderLayer(d_model=args.d_latent, nhead=args.n_head,
                                                     dim_feedforward=args.d_latent, dropout=args.dropout_attn,
                                                     activation=F.relu, layer_norm_eps=1e-8, batch_first=True,
                                                     norm_first=args.norm_first, device=args.device)
        self.encoder = TransformerEncoder(self.encoder_layer, args.n_attn, nn.LayerNorm(args.d_latent, eps=1e-8))

    def forward(self, seq, seq_enc, pos):
        seq_enc += self.pos_emb(pos)
        seq_enc = self.dropout_attn(seq_enc)

        return self.encoder(seq_enc, self.attn_mask, src_key_padding_mask=(seq != self.idx_pad))


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
