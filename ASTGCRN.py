import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MHAtt(nn.Module):
    def __init__(self, input_channels, h=4, q_dim=None, v_dim=None):
        super(MHAtt,self).__init__()
        self.head = h
        q_dim = q_dim or (input_channels // h)
        v_dim = v_dim or (input_channels // h)
        # Query, keys and value matrices
        self._W_q = nn.Linear(input_channels, q_dim*h)  
        self._W_k = nn.Linear(input_channels, q_dim*h)
        self._W_v = nn.Linear(input_channels, v_dim*h)
        # Output linear function
        self._W_o = nn.Linear(h*v_dim, input_channels)

    def forward(self, x):
        """
        [B, C, N, T]
        """
        x = x.permute(0, 2, 3, 1)
        queries = torch.cat(self._W_q(x).chunk(self.head, dim=-1), dim=0)
        keys = torch.cat(self._W_k(x).chunk(self.head, dim=-1), dim=0)
        values = torch.cat(self._W_v(x).chunk(self.head, dim=-1), dim=0)
        # Scaled Dot Product
        D = queries.size(2)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / np.sqrt(D)

        # softmax on the last dimension (v)
        attn_score = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn_score, values)
        # Concatenat the heads
        attention_heads = torch.cat(context.chunk(self.head, dim=0), dim=-1)

        # Apply linear transformation W^O
        out = self._W_o(attention_heads)
        return out.permute(0, 3, 1, 2)


class AGCN(nn.Module):
    """
    Adaptive Graph Convolution
    """
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x: [B, N, dim_in]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
        return x_gconv


class GCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(GCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: (B, num_nodes, input_dim)
        # state: (B, num_nodes, hidden_dim)
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        # print("GCRNCell output shape:", h.shape)
        return h

    def init_hidden_state(self, batch_size):
        return torch.nn.init.xavier_uniform_(torch.empty(batch_size, self.node_num, self.hidden_dim))


class GCRN(nn.Module):
    """
    Graph Convolutional Recurrent Network
    """
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers):
        super(GCRN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(GCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(GCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        # x: (B, T, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        # print(x.shape)
        x = x.permute(0, 3, 2, 1)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
            # print("gcrn layer output shape:", current_inputs.shape)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)


class ASTGCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, seq_len, horizon, num_layers, device):
        super(ASTGCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_layers = num_layers
        self.embed_dim = 10
        self.cheb_k = 3
        self.device = device
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        self.alpha = nn.Parameter(torch.randn(self.cheb_k, 2).to(self.device), requires_grad=True)
        self.encoder = GCRN(self.num_nodes, self.input_dim, self.hidden_dim, self.cheb_k,
                                self.embed_dim, self.num_layers)

        # Three modules based on self-attentive mechanisms
        self.att = MHAtt(self.hidden_dim)
        # self.att = TransformerLayer(args.rnn_units)
        # self.att = InformerLayer(args.rnn_units)
        #predictor
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        # 1x1 convo map from current T to horizon T
        # self.pred_conv = nn.Conv2d(self.seq_len, self.horizon, kernel_size=(1, 1), bias=True)
        self.pred_conv = nn.Conv1d(self.seq_len, self.horizon, kernel_size = 1, bias=True)
        self.end_conv_1 = nn.Conv2d(in_channels=self.hidden_dim,
                                    out_channels=128,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=128,
                                    out_channels=1,
                                    kernel_size=(1, 1),
                                    bias=True)
    
    def forward(self, source):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden

        output = self.att(output.permute(0, 3, 2, 1))  # [B, C, N, T]
        # output = F.leaky_relu(output)
        output = F.leaky_relu(self.end_conv_1(output))
        output = self.end_conv_2(output)  # [B, 1, N, T]
        output = output.permute(0, 3, 2, 1).squeeze(-1)  # [B, T, N]
        # check output has nan
        if torch.isnan(output).any():
            print(output)
        return output[:, :self.horizon, :]