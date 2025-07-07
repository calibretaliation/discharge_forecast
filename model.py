import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([module if i == 0 else type(module)(*module.init_args)
                        for i in range(N)])


class CausalConv1d(nn.Conv1d):
    """
    1-D convolution with padding on the left so that
    the output length equals the input length.

    Expected input shape : (B*N,   C_in, T)
    Output shape         : (B*N,   C_out, T)
    """

    def __init__(self,
                 in_ch:     int,
                 out_ch:    int,
                 kernel_size,
                 dilation=1,
                 **conv_kwargs):
        k = kernel_size[0] if isinstance(kernel_size, (tuple, list)) else kernel_size
        d = dilation[0]    if isinstance(dilation,    (tuple, list)) else dilation

        self.pad_left = (k - 1) * d

        super().__init__(in_ch,
                         out_ch,
                         kernel_size=kernel_size,
                         dilation=dilation,
                         padding=0,         # we pad manually
                         **conv_kwargs)

    def forward(self, x):                  # x: (B*N, C_in, T)
        # F.pad for 1-D expects (left, right)
        x = F.pad(x, (self.pad_left, 0))
        return super().forward(x)          # (B*N, C_out, T)
    
class GatedTemporalConv(nn.Module):
    """
    Gated causal temporal convolution layer
    """

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_sizes=(1, 2, 1),
                 dilations=(1, 2, 1)):
        super().__init__()

        self.in_proj = (nn.Identity()
                        if C_in == C_out
                        else nn.Conv1d(C_in, C_out, kernel_size=1))

        self.convs_gate = nn.ModuleList([
            CausalConv1d(C_out, C_out,
                         kernel_size=k, dilation=d)
            for k, d in zip(kernel_sizes, dilations)
        ])

        self.convs_feat = nn.ModuleList([
            CausalConv1d(C_out, C_out,
                         kernel_size=k, dilation=d)
            for k, d in zip(kernel_sizes, dilations)
        ])

        self.bias_res = nn.Parameter(torch.zeros(1, 1, C_out, 1))

    # ------------------------------------------------------------------
    def _stack(self, x, layers):
        """Pass x through layers *sequentially*."""
        for conv in layers:
            x = conv(x)
        return x

    def forward(self, x):                       # x: (B*N, C_in, T)
        res = self.in_proj(x)                   # (B*N, C_out, T)

        gate_raw = self._stack(res, self.convs_gate)  # (B*N, C_out, T)
        gate     = torch.sigmoid(gate_raw)            # σ(·)

        feat = self._stack(res, self.convs_feat)      # (B*N, C_out, T)

        y = gate * feat + (1.0 - gate) * (res + self.bias_res)

        return y

class STBlock(nn.Module):
    def __init__(self, C_in, C_out, T_in = 30, T_out = 30):
        super().__init__()
        self.temp = GatedTemporalConv(C_in, C_out)
        self.out_channels = C_out * T_out
        self.in_channels = C_out * T_in
        self.T_out = T_out
        # single GATConv that will process all graphs in batch
        self.gat = GATConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            heads=4,
            concat=False,
            dropout=0.3
        )


    def forward(self, x, edge_index=None):
        B, N, C_in, T = x.size()
        # Temporal processing
        x = x.reshape(B*N, C_in, T)  # (B*N, C_in, T)
        h = self.temp(x)  # (B*N,C_out,T)
        h = h.reshape(B, N, -1, T)  # (B,N,C_out,T)
        # Spatial processing
        h_flat = h.reshape(B, N, -1)

        data_list = [
            Data(x=h_flat[b], edge_index=edge_index)
            for b in range(B)
        ]
        batch = Batch.from_data_list(data_list)
        out = self.gat(batch.x, batch.edge_index)
        out, _ = torch_geometric.utils.to_dense_batch(
            out, batch.batch, max_num_nodes=N
        )
        out = out.view(B, N, -1, self.T_out)

        return F.relu(out)


class NhatModel(nn.Module):

    def __init__(self,
                num_nodes: int,
                in_feat: int,
                hid_feat: int = 32,
                num_blocks: int = 3,
                T: int = 30,
                T_hat: int = 10,
                edge_index = None):
        super().__init__()

        C_in = in_feat
        self.blocks = nn.ModuleList(
            [STBlock(in_feat, hid_feat)] +                      # first block
            [STBlock(in_feat, hid_feat) for _ in range(num_blocks - 1)]
        )

        self.res_conv = nn.ModuleList([nn.Conv2d(in_channels=in_feat,
                                                     out_channels=hid_feat,
                                                     kernel_size=(1, 1))] +
                                      [nn.Conv2d(in_channels=hid_feat,
                                                     out_channels=hid_feat,
                                                     kernel_size=(1, 1))for _ in range(num_blocks-1)])
        self.T_hat = T_hat
        self.num_nodes = num_nodes
        self.fc_out = STBlock(hid_feat, 1, T_out = 10)
        #  nn.Sequential(
        #     nn.Linear(30, 30//2),
        #     nn.ReLU(),
        #     nn.Linear(30//2, T_hat)
        # )
        self.edge_index = edge_index
    def forward(self, x):
        # print("STARTING FORWARD")
        # h = x
        x = x.permute(0, 2, 1, 3)  # (B, N, C_in, T)
        all_h = []
        res = x
        # print(x.shape)
        for i in range(len(self.blocks)):
            h = self.blocks[i](x, self.edge_index)
            # print(h.shape)
            res = self.res_conv[i](res.permute(0,2,1,3)).permute(0,2,1,3)
            h = h + res
            # all_h.append(h)
        # all_h = torch.mean(torch.stack(all_h), dim = 0) # (B,N,hid,T)
        # .sum(dim=0)/self.blocks
        y = self.fc_out(h, self.edge_index).squeeze(2)  # (B,N,hid,T)
        # y = y.squeeze(-1).permute(0,2,1)  # (B,N,T_hat)
        return y
