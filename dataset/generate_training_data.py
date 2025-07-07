import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path
from .data_processing import run_data_processing

# Model/Training Config
INPUT_SEQ_LEN = 30
TARGET_SEQ_LEN = 10
NUM_TARGET_FEATURES = 1 # only predict discharge
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
BATCH_SIZE = 32

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SpatioTemporalDataset(Dataset):
    def __init__(self, X, Y):
        """
        Args:
            X (np.array): Input features with shape 
                          (num_samples, input_seq_len, num_nodes, num_features_per_node)
            Y (np.array): Target sequences with shape 
                          (num_samples, target_seq_len, num_nodes, num_target_features)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def load_processed_data():
    """Loads log-transformed timeseries and other processed data."""
    print("Loading processed data...")
    # This file should contain log-transformed but NOT scaled data.
    timeseries_df = np.load(PROCESSED_TIMESERIES_PATH)
    # df_describe = pd.DataFrame(timeseries_df.flatten())
    # print(df_describe.describe())
    static_features = np.load(STATIC_FEATURES_PATH)
    with open(ADJ_MATRIX_PATH, 'rb') as f:
        adj_matrix = pickle.load(f)
        
    print(f"  Timeseries data: {timeseries_df.shape}")
    print(f"  Static features array: {static_features.shape}")
    print(f"  Adjacency matrix shape: {adj_matrix.shape}")
    
    return timeseries_df, static_features, adj_matrix

def scale_data_splits(train_data, val_data, test_data, scaler_save_path):
    """
    Fits scalers on training data for each node and feature individually.
    Uses StandardScaler for discharge (feature 0) and MinMaxScaler for rainfall (feature 1).
    """
    _, num_nodes, num_features = train_data.shape
    
    # all_scalers[0] will be a dict of scalers for discharge {node_idx: scaler}
    # all_scalers[1] will be a dict of scalers for rainfall {node_idx: scaler}
    all_scalers = [{} for _ in range(num_features)]

    scaled_train = np.copy(train_data)
    scaled_val = np.copy(val_data)
    scaled_test = np.copy(test_data)

    print(f"Fitting scalers on training data for {num_nodes} nodes and {num_features} features individually...")

    # Iterate over each feature (e.g., discharge, rainfall)
    for feat_idx in range(num_features):
        if feat_idx == 0: # Feature 0 is discharge
            print("  Scaling discharge (feature 0) using StandardScaler for each node...")
            scaler = StandardScaler()  # Use MinMaxScaler for discharge
        else: # Other features (e.g., rainfall)
            print(f"  Scaling rainfall/other (feature {feat_idx}) for each node...")
            scaler = StandardScaler()
            # continue # Skip rainfall scaling for now
            
        for node_idx in range(num_nodes):
            
            scaler.fit(train_data[:, node_idx, feat_idx].reshape(-1, 1))
            
            # Transform train, val, and test data for this specific node and feature
            scaled_train[:, node_idx, feat_idx] = scaler.transform(train_data[:, node_idx, feat_idx].reshape(-1, 1)).flatten()
            scaled_val[:, node_idx, feat_idx] = scaler.transform(val_data[:, node_idx, feat_idx].reshape(-1, 1)).flatten()
            scaled_test[:, node_idx, feat_idx] = scaler.transform(test_data[:, node_idx, feat_idx].reshape(-1, 1)).flatten()
            
            # Store the fitted scaler for this feature and node
            all_scalers[feat_idx][node_idx] = scaler
        
    print("  Node-and-feature-wise scaling complete.")

    # Save the list of scaler dictionaries
    scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_save_path, "wb") as f:
        pickle.dump(all_scalers, f)
    print(f"  Saved scalers for all time series to {scaler_save_path}")

    return scaled_train, scaled_val, scaled_test

def create_spatio_temporal_sequences(dynamic_data, static_features, 
                                     input_seq_len, target_seq_len, num_target_features=1):
    """Creates sliding window sequences for spatio-temporal forecasting."""
    num_timesteps, num_nodes, num_dynamic_feats = dynamic_data.shape
    if static_features is not None: 
        num_static_feats = static_features.shape[1] 
    else:
        num_static_feats = 0

    dynamic_data_reshaped = dynamic_data.reshape(num_timesteps, num_nodes, num_dynamic_feats)
    
    num_total_input_features = num_dynamic_feats + num_static_feats
    
    X_list, Y_list = [], []
    
    for i in range(num_timesteps - input_seq_len - target_seq_len + 1):
        x_combined = dynamic_data_reshaped[i : i + input_seq_len, :, :]
        static_features_tiled = np.tile(static_features, (input_seq_len, 1, 1))
        if static_features is not None:
            x_combined = np.concatenate((x_combined, static_features_tiled), axis=-1)
        X_list.append(x_combined)
        
        y_slice = dynamic_data_reshaped[i + input_seq_len : i + input_seq_len + target_seq_len, :, :num_target_features]
        Y_list.append(y_slice)
        
    if not X_list:
        return np.array([]).reshape(0, input_seq_len, num_nodes, num_total_input_features), \
               np.array([]).reshape(0, target_seq_len, num_nodes, num_target_features)

    X = np.array(X_list)
    Y = np.array(Y_list)
    
    print(f"Created sequences: X shape {X.shape}, Y shape {Y.shape}")
    return X, Y

def prepare_dataloaders(config=None):
    """
    Main function to load data, split, scale, create sequences, and prepare DataLoaders.
    """
    # Use global config if specific config object is not passed
    input_seq_len = config.seq_len if config and hasattr(config, 'seq_len') else INPUT_SEQ_LEN
    target_seq_len = config.pred_len if config and hasattr(config, 'pred_len') else TARGET_SEQ_LEN
    num_target_features = config.num_target_features if config and hasattr(config, 'num_target_features') else NUM_TARGET_FEATURES
    train_ratio = config.train_ratio if config and hasattr(config, 'train_ratio') else TRAIN_RATIO
    val_ratio = config.val_ratio if config and hasattr(config, 'val_ratio') else VAL_RATIO
    batch_size = config.batch_size if config and hasattr(config, 'batch_size') else BATCH_SIZE

    SCALER_SAVE_PATH = config.PROCESSED_DATA_DIR / "timeseries_node_scalers.pkl" # Path to save the scalers

    _, dynamic_df, static_features, adj_matrix, _ = run_data_processing(save = False,
                                                                        use_all_forcings = config.USE_ALL_FORCINGS, 
                                                                        use_discharge_only = config.USE_DISCHARGE_ONLY, 
                                                                        use_rainfall = config.USE_RAINFALL,
                                                                        input_data_dir = config.INPUT_DATA_DIR, 
                                                                        processed_data_dir = config.PROCESSED_DATA_DIR)
    # dynamic_df, static_features, adj_matrix = load_processed_data()


    if not config.USE_STATIC_FEATURES:
        print("Configuration: Not using static features.")
        static_features = None
    else:
        print("Configuration: Using static features.")
        
    
    # 2. Split data chronologically BEFORE scaling
    num_timesteps = dynamic_df.shape[0]
    train_end_idx = int(num_timesteps * train_ratio)
    val_end_idx = int(num_timesteps * (train_ratio + val_ratio))

    train_df = dynamic_df[:train_end_idx]
    val_df = dynamic_df[train_end_idx:val_end_idx]
    test_df = dynamic_df[val_end_idx:]
    print(f"Data split into Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)}) sets.")

    # 3. Scale data splits (fit on train, transform all) and save scalers
    scaled_train_df, scaled_val_df, scaled_test_df = scale_data_splits(
        train_df, val_df, test_df, SCALER_SAVE_PATH
    )

    print(f"Scaled train data: {scaled_train_df}")
    # 4. Create sequences from each scaled data split
    print("\n--- Creating Training Sequences ---")
    X_train, Y_train = create_spatio_temporal_sequences(
        scaled_train_df, static_features, input_seq_len, target_seq_len, num_target_features
    )

    print("\n--- Creating Validation Sequences ---")
    X_val, Y_val = create_spatio_temporal_sequences(
        scaled_val_df, static_features, input_seq_len, target_seq_len, num_target_features
    )
    print("\n--- Creating Test Sequences ---")
    X_test, Y_test = create_spatio_temporal_sequences(
        scaled_test_df, static_features, input_seq_len, target_seq_len, num_target_features
    )
    
    # 5. Create Datasets and DataLoaders
    train_dataset = SpatioTemporalDataset(X_train, Y_train)
    val_dataset = SpatioTemporalDataset(X_val, Y_val)
    test_dataset = SpatioTemporalDataset(X_test, Y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print("\nDataLoaders created successfully.")
    
    return train_loader, val_loader, test_loader, adj_matrix


def get_adjacency_matrix_and_supports(A, device):
    """Loads a precomputed adjacency matrix and prepares GraphWaveNet style 'supports'."""
    # with open(adj_matrix, 'rb') as f:
    #     A = pickle.load(f)
    
    num_nodes = A.shape[0]
    I = np.eye(num_nodes)
    A_hat = A + I # Add self-loops

    D = A_hat.sum(axis=1)
    D_inv = np.power(D, -1)
    D_inv[np.isinf(D_inv)] = 0.0

    Pf = D_inv.reshape(-1, 1) * A_hat
    Pb = Pf.T
    
    supports = [torch.tensor(Pf, dtype=torch.float32).to(device), 
                torch.tensor(Pb, dtype=torch.float32).to(device)]
    
    print(f"Loaded adjacency matrix and created 'supports' (Pf, Pb) on device: {device}")
    return A, supports


if __name__ == "__main__":
    print("--- Running generate_training_data.py in Debug/Test Mode ---")
    
    import os
    if not os.path.exists(PROCESSED_DATA_DIR):
        print(f"Error: Directory '{PROCESSED_DATA_DIR}' not found. Run data_processing.py first.")
        exit()
    
    train_loader, val_loader, test_loader, adj_m = prepare_dataloaders(config=None)
    
    if train_loader:
        print(f"\nNumber of training batches: {len(train_loader)}")
        x_batch, y_batch = next(iter(train_loader))
        print(f"  Sample X_batch shape: {x_batch.shape}")
        print(f"  Sample Y_batch shape: {y_batch.shape}")
