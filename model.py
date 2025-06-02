import torch
import numpy as np
import torch.nn as nn
import math

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, adj):
        x = self.dropout(self.gc1(x, adj))
        return x


class Dual_BDSTN(nn.Module):
    def __init__(self, S_inputsize, T_inputsize, nhid, outputsize, machine_num, len, dropout, num_layers, nhead):
        super(Dual_BDSTN, self).__init__()
        self.hid_dim = nhid
        self.n_heads = nhead
        assert nhid % nhead == 0
        self.machine_num = machine_num
        self.gcn = GCN(S_inputsize, nhid, dropout)
        # self.self_out = nn.Linear(nhid * self.machine_num, nhid)
        self.gru_emb = self.gru = nn.GRU(T_inputsize, nhid * self.machine_num, num_layers, batch_first=True)
        # Self-Attention
        self.self_w_q = nn.Linear(nhid, nhid)
        self.self_w_k = nn.Linear(nhid, nhid)
        self.self_w_v = nn.Linear(nhid, nhid)
        self.self_scale = np.power(nhid // nhead, 0.5)
        self.self_dropout = nn.Dropout(dropout)
        # self.self_fc = nn.Linear(nhid, nhid)
        self.self_layernorm = nn.LayerNorm([len, nhid * self.machine_num])

        # MultiheadAttention
        self.m_w_q = nn.Linear(nhid, nhid)
        self.m_w_k = nn.Linear(nhid, nhid)
        self.m_w_v = nn.Linear(nhid, nhid)
        self.m_scale = np.power(nhid // nhead, 0.5)
        self.m_dropout = nn.Dropout(dropout)
        # self.m_fc = nn.Linear(nhid, nhid)
        self.m_layernorm = nn.LayerNorm([len, nhid * self.machine_num])

        self.gru = self.gru = nn.GRU(2 * nhid * self.machine_num, nhid, num_layers, batch_first=True)
        self.fc = nn.Linear(nhid, outputsize)

    def forward(self, s, adj, t, mask=None):
        t_out, _ = self.gru_emb(t)
        s = s.reshape(s.shape[0], s.shape[1], 13, -1)
        s_out = self.gcn(s, adj)

        # Self-Attention
        batch_size, len, nhid = t_out.shape
        # Q_t = self.self_w_q(t_out)
        # K_t = self.self_w_k(t_out)
        # V_t = self.self_w_v(t_out)
        Q_t = t_out
        K_t = t_out
        V_t = t_out
        Q_t = Q_t.view(batch_size, len, self.n_heads, self.hid_dim * self.machine_num // self.n_heads).permute(0, 2, 1, 3)
        K_t = K_t.view(batch_size, len, self.n_heads, self.hid_dim * self.machine_num // self.n_heads).permute(0, 2, 1, 3)
        V_t = V_t.view(batch_size, len, self.n_heads, self.hid_dim * self.machine_num // self.n_heads).permute(0, 2, 1, 3)

        self_attn = torch.matmul(Q_t, K_t.permute(0, 1, 3, 2)) / self.self_scale
        if mask is not None:
            self_attn = self_attn.masked_fill(mask == 0, -1e10)
        self_attn = self.self_dropout(torch.softmax(self_attn, dim=-1))
        t_output = torch.matmul(self_attn, V_t)
        t_output = t_output.permute(0, 2, 1, 3).contiguous()
        t_output = t_output.view(batch_size, len, self.n_heads * (self.hid_dim * self.machine_num // self.n_heads))
        # t_output = self.self_fc(t_output) + t_out
        t_output = t_output + t_out
        t_output = self.self_layernorm(t_output)
        # t_output = self.self_out(t_output)

        # MultiheadAttention
        batch_size, len, node, nhid = s_out.shape
        s_out = s_out.reshape(batch_size, len, node * nhid)
        # s_out = self.self_out(s_out)
        # Q_s = self.m_w_q(t_out)
        # K_s = self.m_w_k(s_out)
        # V_s = self.m_w_v(s_out)

        Q_s = t_out
        # Q_s = s_out
        K_s = s_out
        V_s = s_out
        Q_s = Q_s.view(batch_size, len, self.n_heads, self.hid_dim * self.machine_num// self.n_heads).permute(0, 2, 1, 3)
        K_s = K_s.view(batch_size, len, self.n_heads, self.hid_dim * self.machine_num// self.n_heads).permute(0, 2, 1, 3)
        V_s = V_s.view(batch_size, len, self.n_heads, self.hid_dim * self.machine_num// self.n_heads).permute(0, 2, 1, 3)

        m_attn = torch.matmul(Q_s, K_s.permute(0, 1, 3, 2)) / self.m_scale
        if mask is not None:
            m_attn = m_attn.masked_fill(mask == 0, -1e10)
        m_attn = self.m_dropout(torch.softmax(m_attn, dim=-1))
        s_output = torch.matmul(m_attn, V_s)
        s_output = s_output.permute(0, 2, 1, 3).contiguous()
        s_output = s_output.view(batch_size, len, self.n_heads * (self.hid_dim * self.machine_num// self.n_heads))
        # s_output = self.m_layernorm(self.m_fc(s_output)+s_out)
        s_output = self.m_layernorm(s_output + s_out)
        # Concat
        output = torch.cat((s_output, t_output), dim=-1)
        # gru
        output, _ = self.gru(output)
        # output
        output = self.fc(output)
        return output[:, -1, :].view(output.shape[0], -1, output.shape[2])

if __name__ == "__main__":
   batch_size, len, node, d = 64, 8, 13, 2
   nhid = 64
   nhead = 1
   dropout = 0.8
   num_layers = 2
   outputsize = node * 2
   s = torch.randn(batch_size, len, node, d)
   t = torch.randn(batch_size, len, node * 2)
   adj_dynamic = torch.randn(batch_size, len, node, node)
   dual_bdstn = Dual_BDSTN(d, node * 2, nhid, outputsize, node, len, dropout, num_layers, nhead)
   output = dual_bdstn(s, adj_dynamic, t)
   print(output.shape)