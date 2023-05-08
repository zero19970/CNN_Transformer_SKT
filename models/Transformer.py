import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn, Tensor

class LearnableGlobalLocalMultiheadAttention(nn.Module):
    NUM_WEIGHTS = 9
    def __init__(
            self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(self.NUM_WEIGHTS * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(self.NUM_WEIGHTS * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.bias_k = self.bias_v = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)


    # global
    def in_proj_global_q(self, query):
        return self._in_proj(query, start=0, end=self.embed_dim)

    def in_proj_global_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_global_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim, end=3 * self.embed_dim)

    # local left
    def in_proj_local_left_q(self, query):
        return self._in_proj(query, start=3 * self.embed_dim, end=4 * self.embed_dim)

    def in_proj_local_left_k(self, key):
        return self._in_proj(key, start=4 * self.embed_dim, end=5 * self.embed_dim)

    # local right
    def in_proj_local_right_q(self, query):
        return self._in_proj(query, start=5 * self.embed_dim, end=6 * self.embed_dim)

    def in_proj_local_right_k(self, key):
        return self._in_proj(key, start=6 * self.embed_dim, end=7 * self.embed_dim)

    # local right
    def in_proj_local_q(self, query):
        return self._in_proj(query, start=7 * self.embed_dim, end=8 * self.embed_dim)

    def in_proj_local_k(self, key):
        return self._in_proj(key, start=8 * self.embed_dim, end=9 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)



    def prepare_local_masking(self, q_left, k_left, q_right, k_right, shape):

        left_attn_weights = torch.bmm(q_left, k_left.transpose(1, 2))
        right_attn_weights = torch.bmm(q_right, k_right.transpose(1, 2))

        left_size = left_attn_weights.size()
        src_len = left_size[2]

        triu = torch.ones(src_len, src_len, device=q_left.device, dtype=q_left.dtype).triu_()
        mini_triu = torch.ones(shape[1], shape[1], device=q_left.device, dtype=q_left.dtype).triu_()
        mini_triu = mini_triu.repeat(shape[0], shape[0])
        triu = (triu * mini_triu).unsqueeze_(0)

        left_softmax = F.softmax(left_attn_weights, dim=-1)
        right_softmax = F.softmax(right_attn_weights, dim=-1)

        local_mask = self.compute_lrmask2localmask(left_softmax, right_softmax, triu)

        return local_mask

    def compute_lrmask2localmask(self, left_softmax, right_softmax, triu):
        triu_t = triu.transpose(1,2)
        left_mask = torch.matmul(left_softmax, triu)
        right_mask = torch.matmul(right_softmax, triu_t)
        bw_left_mask = torch.matmul(left_softmax, triu_t)
        bw_right_mask = torch.matmul(right_softmax, triu)

        fw_mask = left_mask * right_mask
        bw_mask = bw_left_mask * bw_right_mask
        local_mask = fw_mask + bw_mask
        return local_mask

    def forward(self, query, key, shape, value):

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        q = self.in_proj_global_q(query)
        k = self.in_proj_global_k(key)
        v = self.in_proj_global_v(value)
        q_left = self.in_proj_local_left_q(query)
        k_left = self.in_proj_local_left_k(key)
        q_right = self.in_proj_local_right_q(query)
        k_right = self.in_proj_local_right_k(key)
        q_local = self.in_proj_local_q(query)
        k_local = self.in_proj_local_k(key)

        q = q*self.scaling
        q_local = q_local * self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q_local = q_local.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k_local = k_local.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k_left = k_left.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k_right = k_right.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q_left = q_left.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q_right = q_right.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        global_attn_weights = torch.bmm(q, k.transpose(1, 2))
        local_attn_weights = torch.bmm(q_local, k_local.transpose(1, 2))

        local_att_mask = self.prepare_local_masking(q_left, k_left, q_right, k_right, shape)
        masked_local_attn_weights = local_attn_weights * local_att_mask

        attn_weights = 0.1 * global_attn_weights + masked_local_attn_weights

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        consistent_mask = torch.sum(local_att_mask, dim=0)

        return attn, consistent_mask

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, shape,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        features = []

        for layer in self.layers:
            output, consistent_feature = layer(output, shape, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            features.append(consistent_feature)

        if self.norm is not None:
            output = self.norm(output)

        return output, features

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = LearnableGlobalLocalMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src, shape,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)

        src2, mask = self.self_attn(q, k, shape, src)
        feature = torch.squeeze(src, dim=1)
        consistent_feature = torch.matmul(mask, feature)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, consistent_feature

    def forward_pre(self, src, shape,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        ssrc2, mask = self.self_attn(q, k, shape, src)
        feature = torch.squeeze(src, dim=1)
        consistent_feature = torch.matmul(mask, feature)

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, consistent_feature

    def forward(self, src, shape,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, shape, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, shape, src_mask, src_key_padding_mask, pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")