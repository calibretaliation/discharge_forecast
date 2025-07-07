from datetime import datetime
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

from utils import *


class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        """
        Args:
            X (np.array): Input features with shape (num_samples, num_nodes, input_seq_len)
            Y (np.array): Target sequences with shape (num_samples, num_nodes, target_seq_len)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        # Returns a tuple (input_features, target_sequence) for the given index
        return self.X[idx], self.Y[idx]

def prepare_data(config, graph_filename = "usgs_graph_filtered_discharge.gpickle"):
    print("LOAD GRAPH")
    with open(graph_filename, 'rb') as f:
        G = pickle.load(f)
        
    train_start = datetime(2020, 1, 1)
    train_end   = datetime(2022, 10, 31)  
    val_start   = datetime(2023, 1, 1)
    val_end     = datetime(2024, 12, 31)

    input_seq_len = config.seq_len
    target_seq_len = config.pred_len
    print("BUILD NODE FEATURES")

    
    train_tensor, train_dates, nodes_order = build_node_feature_tensor(G, train_start, train_end, config)
    val_tensor,   val_dates,   _          = build_node_feature_tensor(G, val_start, val_end, config)
    config.num_nodes  = train_tensor.shape[0]
    config.num_feats  = train_tensor.shape[1]   # 4

    N, F, T = train_tensor.shape
    print(config.num_nodes, "nodes in the graph")

    scaler = NodeStandardScaler3D().fit(train_tensor)
    scaler.save("node_scaler.pt") 
    
    train_tensor = scaler.transform(train_tensor)
    val_tensor   = scaler.transform(val_tensor)
    
    # Build the train and validation datasets using sliding windows  
    X_train, Y_train = build_dataset(train_tensor, input_seq_len, target_seq_len)
    X_val,   Y_val   = build_dataset(val_tensor,   input_seq_len, target_seq_len)
    print("Training samples shape:", X_train.shape)
    A = build_adjacency_matrix(G, nodes_order)

    I = torch.eye(config.num_nodes)
    A_hat = torch.Tensor(A) + I

    # Compute D = sum of each row
    D = A_hat.sum(dim=1)                 # shape [N]
    D_inv = D.pow(-1)                    # shape [N]
    D_inv[torch.isinf(D_inv)] = 0.0      # replace inf with 0

    # Pf = D^-1 A_hat
    Pf = D_inv.view(-1, 1) * A_hat       # shape [N, N]
    Pf = Pf.to(config.device)

    # Pb = (Pf)^T
    Pb = Pf.t().to(config.device)

    edge_index, _ = adj_to_edge_index_and_attr(A)
    edge_index = edge_index.to(config.device)

    print("Training dataset shapes:")
    print("  X_train:", X_train.shape)  # (num_samples_train, num_nodes, input_seq_len)
    print("  Y_train:", Y_train.shape)  # (num_samples_train, num_nodes, target_seq_len)
    print("Validation dataset shapes:")
    print("  X_val:", X_val.shape)      # (num_samples_val, num_nodes, input_seq_len)
    print("  Y_val:", Y_val.shape)      # (num_samples_val, num_nodes, target_seq_len)
    print("Adjacency matrix shape:", A.shape)  # (num_nodes, num_nodes)

    train_dataset = TimeSeriesDataset(X_train, Y_train)
    val_dataset = TimeSeriesDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    supports = [Pf, Pb]
    return train_loader, val_loader, supports, edge_index, A