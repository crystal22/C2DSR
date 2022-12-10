import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import torch.nn.functional as F


class SelfAttention(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.device
        self.d_latent = args.d_latent
        self.len_max = args.len_max
        self.n_head = args.n_head
        self.dropout = args.dropout
        self.n_attn = args.n_attn
        self.norm_first = args.norm_first
        self.idx_pad = args.idx_pad

        attn_mask = ~torch.tril(torch.ones((self.len_max, self.len_max), dtype=torch.bool))
        self.register_buffer('attn_mask', attn_mask)

        self.emb_dropout = nn.Dropout(p=self.dropout)
        self.pos_emb = nn.Embedding(self.len_max, self.d_latent)  # TO IMPROVE

        # loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.encoder_layer = TransformerEncoderLayer(d_model=self.d_latent, nhead=self.n_head,
                                                     dim_feedforward=self.d_latent, dropout=self.dropout,
                                                     activation=F.relu, layer_norm_eps=1e-8, batch_first=True,
                                                     norm_first=self.norm_first, device=self.device)
        self.encoder = TransformerEncoder(self.encoder_layer, self.n_attn, nn.LayerNorm(self.d_latent, eps=1e-8))

    def forward(self, seq, seq_enc, pos):
        seq_enc += self.pos_emb(pos)
        seq_enc = self.emb_dropout(seq_enc)

        key_padding_mask = (seq == self.idx_pad).bool()

        return self.encoder(seq_enc, self.attn_mask, key_padding_mask)
