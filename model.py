import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv

class STBlock(nn.Module):
    def __init__(self, in_channels, spatial_hidden, num_heads, edge_index):
        super().__init__()
        self.in_channels = in_channels
        # Spatial layer: GATConv
        self.gat = GATConv(in_channels, spatial_hidden, heads=num_heads)
        C_hidden = spatial_hidden * num_heads
        self.bn_spatial = nn.BatchNorm1d(C_hidden)
        # Temporal layer: LSTM
        self.lstm = nn.LSTM(C_hidden, in_channels, num_layers=2, batch_first=False)
        self.bn_temporal = nn.BatchNorm1d(in_channels)
        self.edge_index = edge_index

    def forward(self, x):
        B, C, N, T = x.shape
        device = x.device
        # Create batched edge_index for all graphs in the batch
        batched_edge_index = torch.cat([self.edge_index + b * N for b in range(B)], dim=1).to(device)
        
        # Spatial processing with GATConv for each time step
        h_list = []
        for t in range(T):
            x_t = x[:, :, :, t].reshape(B * N, C)
            h_t = self.gat(x_t, batched_edge_index)
            h_t = h_t.reshape(B, N, -1)  # (B, N, C_hidden)
            h_t = self.bn_spatial(h_t.permute(0, 2, 1)).permute(0, 2, 1)  # (B, N, C_hidden)
            h_list.append(h_t)
        h = torch.stack(h_list, dim=1)  # (B, T, N, C_hidden)
        
        # Temporal processing with LSTM
        h = h.permute(1, 0, 2, 3).reshape(T, B * N, -1)  # (T, B*N, C_hidden)
        lstm_out, _ = self.lstm(h)  # (T, B*N, in_channels)
        lstm_out = self.bn_temporal(lstm_out.permute(1, 2, 0)).permute(2, 0, 1)  # (T, B*N, in_channels)
        lstm_out = lstm_out.view(T, B, N, self.in_channels).permute(1, 3, 2, 0)  # (B, in_channels, N, T)
        
        # Residual connection
        out = x + lstm_out
        return out

class NhatModelBlock(nn.Module):
    def __init__(self, edge_index, config, num_blocks):
            super().__init__()
            self.edge_index = edge_index
            self.config = config
            # Stack multiple STBlocks
            self.st_blocks = nn.ModuleList([
                STBlock(self.config.in_feat, self.config.hid_feat, 4, edge_index) 
                for _ in range(num_blocks)
            ])
            # Final LSTM for sequence processing
            self.final_lstm = nn.LSTM(self.config.in_feat, self.config.hid_feat, num_layers=1, batch_first=False)
            # Linear layer to predict 10 future steps
            self.linear = nn.Linear(self.config.hid_feat, self.config.pred_len)

    def forward(self, x):
        B, C, N, T = x.shape
        
        # Pass input through all STBlocks
        for block in self.st_blocks:
            x = block(x)  # (B, C, N, T)
        
        # Prepare for final LSTM
        x = x.permute(3, 0, 2, 1).reshape(T, B * N, C)  # (T, B*N, C)
        _, (hidden, _) = self.final_lstm(x)  # hidden: (1, B*N, pred_hidden)
        hidden = hidden[0]  # (B*N, pred_hidden)
        
        # Predict next 10 time steps
        pred = self.linear(hidden)  # (B*N, pred_len)
        pred = pred.view(B, N, self.config.pred_len).permute(0, 2, 1)  # (B, pred_len, N)
        return pred