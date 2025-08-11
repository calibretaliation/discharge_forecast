import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch_geometric.nn import GATConv

from gat import GATLayerImp3


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    
class temporalAttention(nn.Module):
    def __init__(self, d_model, d_k, heads):
        super(temporalAttention, self).__init__()
        self.d_k = d_k
        self.query = nn.Linear(d_model, d_k * heads)
        self.key = nn.Linear(d_model, d_k * heads)
        self.value = nn.Linear(d_model, d_k * heads)
        self.softmax = nn.Softmax(dim=-1)
        self.heads = heads
        self.linear = nn.Sequential(
            nn.Linear(d_k * heads, d_k*heads//2),
            nn.ReLU(),
            nn.Linear(d_k * heads//2, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self, x):
        batch_size, _, num_nodes, seq_len = x.size()
        x = x.permute(0, 2, 3, 1)  # Change to (batch_size, num_nodes, seq_len, d_model)
        # x = x.view(batch_size * num_nodes, seq_len, d_k)
        # print("attention input: ", x.shape)
        query = self.query(x).view(batch_size, num_nodes, seq_len, self.heads, self.d_k).permute(0, 1, 3, 2, 4) # (batch_size, num_nodes, heads, seq_len, d_k)
        key = self.key(x).view(batch_size, num_nodes, seq_len, self.heads, self.d_k).permute(0, 1, 3, 2, 4)
        value = self.value(x).view(batch_size, num_nodes, seq_len, self.heads, self.d_k).permute(0, 1, 3, 2, 4)
        # print("query: ", query.shape)
        # print("key transposed: ", key.transpose(-2, -1).shape)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)  # (batch_size, num_nodes, heads, seq_len, seq_len)
        attention_weights = self.softmax(attention_scores)
        # print("attention weights: ", attention_weights.shape)

        context = torch.matmul(attention_weights, value)  # (batch_size, num_nodes, heads, seq_len, d_k)
        # print("context: ", context.shape)
        context = context.permute(0, 1, 3, 2, 4).contiguous()
        # print("context: ", context.shape)
        context = context.view(batch_size, num_nodes, seq_len, self.heads * self.d_k)
        # context = context.mean(dim=3)  # (batch_size, num_nodes, heads, d_k)
        # print("context: ", context.shape)
        # ADD + NORM
        context = self.layer_norm(x + self.linear(context))
        context = context.permute(0, 3, 1, 2) # (batch_size, d_k , num_nodes, seq_len)
        # print("context: ", context.shape)
        return context

class gat(nn.Module):
    """
    A GAT layer adapted for spatio-temporal inputs with shape (B, C, N, T),
    where B is batch size, C is input features, N is number of nodes, T is time steps.
    The graph structure (edge_index) is assumed to be static across batches and time steps.
    Outputs shape (B, C_out, N, T), where C_out = num_out_features * num_of_heads if concat else num_out_features.
    """
    def __init__(self, num_in_features, num_out_features, num_of_heads=4, concat=True, activation=nn.ELU(),
                 dropout=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):
        super().__init__()
        self.gat_layer = GATLayerImp3(
            num_in_features=num_in_features,
            num_out_features=num_out_features,
            num_of_heads=num_of_heads,
            concat=concat,
            activation=activation,
            dropout_prob=dropout,
            add_skip_connection=add_skip_connection,
            bias=bias,
            log_attention_weights=log_attention_weights
        )

    def forward(self, in_nodes_features, edge_index):
        """
        :param in_nodes_features: torch.Tensor with shape (B, C, N, T)
        :param edge_index: torch.Tensor with shape (2, E), sparse edge index for the graph
        :return: torch.Tensor with shape (B, C_out, N, T)
        """
        B, C, N, T = in_nodes_features.shape
        K = B * T
        total_nodes = K * N

        # Reshape features to (total_nodes, C)
        features = in_nodes_features.permute(0, 3, 2, 1).reshape(total_nodes, C)

        # Create batched edge_index with offsets
        E = edge_index.shape[1]
        device = edge_index.device
        batched_edge_index = edge_index.unsqueeze(0).repeat(K, 1, 1)  # (K, 2, E)
        offsets = torch.arange(K, device=device) * N
        offsets = offsets.view(K, 1, 1).expand(K, 2, E)  # (K, 2, E)
        batched_edge_index += offsets
        batched_edge_index = batched_edge_index.reshape(2, K * E)  # (2, K*E)

        # Forward through the underlying GAT layer
        data = (features, batched_edge_index)
        out_features, _ = self.gat_layer(data)  # (total_nodes, C_out)

        # Reshape back to (B, C_out, N, T)
        C_out = out_features.shape[1]
        out_features = out_features.view(B, T, N, C_out).permute(0, 3, 2, 1)

        return out_features
    
class gwnet_new(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=128,end_channels=128,
                 kernel_size=2,blocks=4,layers=2):
        super(gwnet_new, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs_1 = nn.ModuleList()
        self.gate_convs_1 = nn.ModuleList()
        self.filter_convs_2 = nn.ModuleList()
        self.gate_convs_2 = nn.ModuleList()
        self.filter_convs_3 = nn.ModuleList()
        self.gate_convs_3 = nn.ModuleList()
        self.tcn_fusion = nn.ModuleList()

        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.temporal_attentions = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports
        self.new_supports = None
        receptive_field = 1
        self.dilations = []
        self.supports_len = 0
        self.kernel_sizes = [2,4,8]
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.dilations.append(new_dilation)
                self.temporal_attentions.append(temporalAttention(d_model=residual_channels, d_k=8, heads=4))
                # dilated convolutions 1
                self.filter_convs_1.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,self.kernel_sizes[0]),dilation=new_dilation))

                self.gate_convs_1.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, self.kernel_sizes[0]), dilation=new_dilation))

                # dilated convolutions 2
                self.filter_convs_2.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,self.kernel_sizes[1]),dilation=new_dilation))

                self.gate_convs_2.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, self.kernel_sizes[1]), dilation=new_dilation))

                # dilated convolutions 3
                self.filter_convs_3.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,self.kernel_sizes[2]),dilation=new_dilation))

                self.gate_convs_3.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, self.kernel_sizes[2]), dilation=new_dilation))

                self.tcn_fusion.append(nn.Conv2d(in_channels=dilation_channels * 3, 
                                                 out_channels=dilation_channels, 
                                                 kernel_size=1))
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout=dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field


    def forward(self, input):
        # in_len = input.size(3)
        # if in_len<self.receptive_field:
        #     x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        # else:
        x = input
        # print("START SHAPE: ", x.shape)
        x = self.start_conv(x)
        # print("PROJECTED INPUT: ", x.shape)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        self.new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            self.new_supports = self.supports + [adp]
            # self.new_supports[0] = torch.ones_like(self.new_supports[0])/len(self.new_supports[0])

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)

            residual = x
            x = self.temporal_attentions[i](x)  # Apply temporal attention
            # print("X before: ", x.shape)

            # dilated convolution 1
            pad_amount = self.dilations[i] * (self.kernel_sizes[0] - 1)
            residual_1 = F.pad(residual, (pad_amount, 0))
            filter = self.filter_convs_1[i](residual_1)
            # print("Filter: ", filter.shape)
            filter = torch.tanh(filter)
            gate = self.gate_convs_1[i](residual_1)
            gate = torch.sigmoid(gate)
            # print("Gate: ", gate.shape)
            x_1 = filter * gate

            # dilated convolution 2
            pad_amount = self.dilations[i] * (self.kernel_sizes[1] - 1)
            residual_2 = F.pad(residual, (pad_amount, 0))
            filter = self.filter_convs_2[i](residual_2)
            # print("Filter: ", filter.shape)
            filter = torch.tanh(filter)
            gate = self.gate_convs_2[i](residual_2)
            gate = torch.sigmoid(gate)
            # print("Gate: ", gate.shape)
            x_2 = filter * gate

            # dilated convolution 3
            pad_amount = self.dilations[i] * (self.kernel_sizes[2] - 1)
            residual_3 = F.pad(residual, (pad_amount, 0))
            filter = self.filter_convs_3[i](residual_3)
            # print("Filter: ", filter.shape)
            filter = torch.tanh(filter)
            gate = self.gate_convs_3[i](residual_3)
            gate = torch.sigmoid(gate)
            # print("Gate: ", gate.shape)
            x_3 = filter * gate

            x = torch.cat([x_1, x_2, x_3], dim=1)
            x = self.tcn_fusion[i](x)
            # print("AFTER TCN: ", x.shape)

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            # print("AFTER SKIP: ", x.shape)
            B, C, N, T = x.shape
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    # convert new_supports to edge_index format for GATConv
                    edge_index = self.new_supports[0].nonzero(as_tuple=False).t().contiguous().to(x.device)
                    x = self.gconv[i](x, self.new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x[..., -1]

    def get_learned_adj(self):
        return self.new_supports

