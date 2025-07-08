# import sys, os
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import sys
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
# --- Import from our custom modules ---
# Assuming these files are in the same directory
from dataset.generate_training_data import prepare_dataloaders, get_adjacency_matrix_and_supports
from graph_wavenet import gwnet
from model import NhatModel 

# --- Configuration ---
class TrainingConfig:
    # Data and Model Config
    seq_len = 30                # Input sequence length (must match generate_training_data.py)
    pred_len = 10               # Target sequence length (must match generate_training_data.py)
    num_target_features = 1     # We predict only discharge
    
    # Training Hyperparameters
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    batch_size = 32
    learning_rate = 1e-3
    
    # Paths
    PROCESSED_DATA_DIR = Path("dataset/processed_camels_discharge_only")
    # PRETRAIN_MODEL_PATH = "weights/camels/best_gwnet_model.pt"
    PRETRAIN_MODEL_PATH = "None" #Train from scratch
    BEST_MODEL_SAVE_PATH = "weights/camels/best_gwnet_model.pt"
    PLOT_SAVE_DIR = Path("plots/training/discharge_only")
    INPUT_DATA_DIR = "dataset/data_camels"
    SCALER_PATH = PROCESSED_DATA_DIR / "timeseries_node_scalers.pkl"
    ADJ_MATRIX_PATH = PROCESSED_DATA_DIR / "adjacency_matrix.pkl"
    NEW_ADJ_MATRIX_PATH = PROCESSED_DATA_DIR / "new_adjacency_matrix.pkl"
    PROCESSED_DISCHARGE_PATH = PROCESSED_DATA_DIR / "processed_discharge.csv"
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15

    USE_RAINFALL = False         # Set to False to use only 1 time series (discharge)
    USE_STATIC_FEATURES = False   # Set to False to exclude static basin attributes
    USE_ALL_FORCINGS = False
    USE_DISCHARGE_ONLY = True
    USE_GRAPH = False

def unscale_data(data, scaler_dict):
    """
    Inverse transforms the scaled data back to its original scale using per-node scalers.
    
    Args:
        data (torch.Tensor): Scaled data of shape (batch_size, num_nodes, pred_len) or similar.
        scaler_dict (dict): Dictionary where keys are node IDs and values are fitted StandardScaler objects.
    
    Returns:
        torch.Tensor: Data in its original scale.
    """
    data = data.cpu().detach().numpy()
    unscaled_data = np.zeros_like(data)
    
    node_order = list(scaler_dict.keys()) # Assumes scaler_dict keys are in the correct order
    batch_size, num_nodes, pred_len = data.shape
    
    # Simpler loop for clarity in evaluation:
    for i in range(num_nodes):
        node_id = node_order[i]
        scaler = scaler_dict[node_id]
        for j in range(batch_size):
            # Squeeze to make it (pred_len,) -> (pred_len, 1) for scaler
            node_ts = data[j, i, :].reshape(-1, 1) 
            unscaled_data[j, i, :] = scaler.inverse_transform(node_ts).flatten()

    return torch.from_numpy(unscaled_data)

def calculate_nse(y_true, y_pred):
    """Calculates Nash-Sutcliffe Efficiency."""
    numerator = torch.sum((y_true - y_pred)**2)
    denominator = torch.sum((y_true - torch.mean(y_true))**2)
    if denominator == 0:
        print("y_true:", y_true)
        print("y_pred:", y_pred)
        print("numerator:", numerator)
        print("denominator:", denominator)
        return -float('inf')
    nse = 1 - (numerator / denominator)
    # print when nse too high or infinity
    if (abs(nse) > 1e4)  or (nse == float('inf')) or (nse == -float('inf')):
        print("Warning: NSE value is unusually high. This may due to flat y_true value.")
        print("y_true:", y_true)
        print("y_pred:", y_pred)
        print("numerator:", numerator)
        print("denominator:", denominator)
    return nse.item()


class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, config, supports, scaler, site_order):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.supports = supports
        self.scaler = scaler[0] # Dictionary of scalers for inverse transform
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_nse_history = []
        self.site_order = site_order

    def plot_metrics(self):
            """Plots and saves the training and validation metrics."""
            print("Plotting training metrics...")
            epochs = range(1, len(self.train_loss_history) + 1)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

            # Plot Loss
            ax1.plot(epochs, self.train_loss_history, 'g-o', label='Training Loss')
            ax1.plot(epochs, self.val_loss_history, 'b-o', label='Validation Loss')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)

            # Plot NSE
            ax2.plot(epochs, self.val_nse_history, 'r-o', label='Validation NSE')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('NSE')
            ax2.set_title('Validation Nash-Sutcliffe Efficiency')
            ax2.legend()
            ax2.grid(True)
            ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.8) # Add y=0 line for reference

            plt.tight_layout()
            save_path = self.config.PLOT_SAVE_DIR / "training_progress.png"
            plt.savefig(save_path)
            print(f"Saved training progress plot to {save_path}")
            plt.close()

    def plot_sample_prediction(self, epoch):
        """Plots a single sample prediction against the ground truth."""
        print("Plotting sample prediction...")
        self.model.eval()
        with torch.no_grad():
            x_batch, y_batch = next(iter(self.val_loader))
            x_batch = x_batch.to(self.config.device)
            
            x_permuted = x_batch.permute(0, 3, 2, 1)
            output = self.model(x_permuted).squeeze(1)

            # Choose a random sample and node from the batch to plot
            sample_idx = np.random.randint(0, x_batch.size(0))
            node_idx = np.random.randint(0, x_batch.size(2))
            # site_id = self.site_order[node_idx]
            # Get the history and ground truth for the selected sample and node
            # The model predicts based on scaled data, so we need to unscale for plotting
            history_scaled = x_batch[sample_idx, :, node_idx, 0] # Feature 0 is discharge
            truth_scaled = y_batch[sample_idx, :, node_idx, 0]
            # print(output.shape)
            # print(x_batch.shape)
            pred_scaled = output[sample_idx, :, node_idx]

            # Unscale for plotting
            discharge_scaler_dict = self.scaler
            # print(discharge_scaler_dict)
            truth_unscaled = unscale_data(truth_scaled.unsqueeze(0).unsqueeze(0), {0: discharge_scaler_dict[node_idx]})[0,0,:]
            pred_unscaled = unscale_data(pred_scaled.unsqueeze(0).unsqueeze(0), {0: discharge_scaler_dict[node_idx]})[0,0,:]
            history_unscaled = unscale_data(history_scaled.unsqueeze(0).unsqueeze(0), {0: discharge_scaler_dict[node_idx]})[0,0,:]

            # Reverse the log transform
            # truth_final = torch.expm1(truth_unscaled)
            # pred_final = torch.expm1(pred_unscaled)
            # history_final = torch.expm1(history_unscaled)

            plt.figure(figsize=(15, 6))
            total_len = len(history_unscaled) + len(truth_unscaled)
            # Plot history
            plt.plot(np.arange(0, len(history_unscaled)), history_unscaled.cpu(), label='Input History', color='gray')
            # Plot ground truth
            plt.plot(np.arange(len(history_unscaled), total_len), truth_unscaled.cpu(), label='Ground Truth', color='blue', marker='o')
            # Plot prediction
            plt.plot(np.arange(len(history_unscaled), total_len), pred_unscaled.cpu(), label='Prediction', color='red', linestyle='--')

            plt.title(f'Sample Prediction vs. Ground Truth (Epoch {epoch}, Node {node_idx})')
            plt.xlabel('Time Steps')
            plt.ylabel('Discharge')
            plt.legend()
            plt.grid(True)
            save_path = self.config.PLOT_SAVE_DIR / f"prediction_epoch_{epoch}.png"
            plt.savefig(save_path)
            plt.close()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for x_batch, y_batch in tqdm(self.train_loader, desc="Training", leave=False):
            x_batch = x_batch.to(self.config.device)
            y_batch = y_batch.to(self.config.device)

            self.optimizer.zero_grad()
            
            # Dataloader provides: (batch, seq_len, nodes, features)
            # Model's Conv2d expects: (batch, features, nodes, seq_len)
            x_permuted = x_batch.permute(0, 3, 2, 1)
            # print("node sample: ", x_permuted[0,:,random.randint(0, x_permuted.shape[2] - 1),0])
            output = self.model(x_permuted)     
            output = output.squeeze(1) # -> (batch_size, pred_len, num_nodes)
            # Target from dataloader is (batch, pred_len, nodes, 1)
            # Squeeze the last dimension to match model output
            y_target = y_batch.squeeze(-1) # -> (batch_size, pred_len, num_nodes)
            loss = self.criterion(output, y_target)

            loss.backward()
            
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)
        
    
    def eval_epoch(self):
        self.model.eval()
        total_loss = 0
        all_y_true_unscaled = []
        all_y_pred_unscaled = []
        # print(self.scaler)
        with torch.no_grad():
            for x_batch, y_batch in tqdm(self.val_loader, desc="Validating", leave=False):
                x_batch = x_batch.to(self.config.device)
                y_batch = y_batch.to(self.config.device)
                x_permuted = x_batch.permute(0, 3, 2, 1)
                output = self.model(x_permuted).squeeze(1)
                y_target = y_batch.squeeze(-1)

                loss = self.criterion(output, y_target)
                if not torch.isnan(loss):
                    total_loss += loss.item()
                
                # unscale_data expects (batch, nodes, pred_len)
                output_unscaled = unscale_data(output, self.scaler)
                y_target_unscaled = unscale_data(y_target, self.scaler)
                
                all_y_pred_unscaled.append(output_unscaled)
                all_y_true_unscaled.append(y_target_unscaled) # Keep y_target in (batch, nodes, pred_len) format

        all_y_true = torch.cat(all_y_true_unscaled, dim=0)
        all_y_pred = torch.cat(all_y_pred_unscaled, dim=0)
        
        # all_y_true_unlogged = torch.expm1(all_y_true)
        # all_y_pred_unlogged = torch.expm1(all_y_pred)

        num_nodes = all_y_true.shape[1]
        batch_size = all_y_true.shape[0]
        nse_per_node = []
        for i in range(num_nodes):
            for j in range(batch_size):
                y_true_node_i = all_y_true[j, i, :].flatten()
                y_pred_node_i = all_y_pred[j, i, :].flatten()
                node_nse = calculate_nse(y_true_node_i, y_pred_node_i)
                nse_per_node.append(node_nse)
        valid_nses = [n for n in nse_per_node if abs(n) < 1e4]  # Filter out extreme NSE values
        average_nse = np.mean(valid_nses) if valid_nses else -np.inf

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss, average_nse

    def train(self):
        self.config.PLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        Path(self.config.BEST_MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        best_val_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            avg_train_loss = self.train_epoch()
            avg_val_loss, val_nse = self.eval_epoch()

            self.train_loss_history.append(avg_train_loss)
            self.val_loss_history.append(avg_val_loss)
            self.val_nse_history.append(val_nse)

            print(f"Epoch {epoch}/{self.config.num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val NSE: {val_nse:.4f}")

            if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss):
                best_val_loss = avg_val_loss
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                torch.save(self.model.state_dict(), self.config.BEST_MODEL_SAVE_PATH)
                new_adj = self.model.get_learned_adj()[0]
                if new_adj is not None:
                    print("Learned adjacency matrix shape:", new_adj[:10,:10])
                    # Save the learned adjacency matrix for inspection
                    with open(self.config.NEW_ADJ_MATRIX_PATH, 'wb') as f:
                        pickle.dump(new_adj.cpu().detach().numpy(), f)
                    print(f"Saved learned adjacency matrix to {self.config.NEW_ADJ_MATRIX_PATH}")
                    # self.plot_sample_prediction(epoch)
            self.plot_metrics()



def pre_run_diagnostics(config):
    """
    Performs checks on the data and graph before starting the main training loop.
    Returns True if checks pass, False otherwise.
    """
    print("\n--- Running Pre-run Diagnostics ---")
    success = True

    try:
        with open(config.ADJ_MATRIX_PATH, 'rb') as f:
            adj_matrix = pickle.load(f)
        if np.any(np.isnan(adj_matrix)):
            print("🛑 DIAGNOSTIC FAILED: Adjacency matrix contains NaN values.")
            success = False
        degrees = adj_matrix.sum(axis=1)
        isolated_nodes = np.where(degrees == 0)[0]
        if len(isolated_nodes) > 0:
            print(f"⚠️ DIAGNOSTIC WARNING: Found {len(isolated_nodes)} isolated nodes (degree 0).")
    except FileNotFoundError:
        print(f"🛑 DIAGNOSTIC FAILED: Adjacency matrix not found at {config.ADJ_MATRIX_PATH}")
        success = False

    try:
        # --- FIX IS HERE ---
        # Correctly load the CSV by specifying the first column as the index.
        df = pd.read_csv(config.PROCESSED_DISCHARGE_PATH, index_col=0, parse_dates=True)
        
        if df.isnull().values.any():
            print("🛑 DIAGNOSTIC FAILED: The main data file contains NaN values AFTER processing.")
            success = False
        
        num_timesteps = df.shape[0]
        train_end_idx = int(num_timesteps * config.TRAIN_RATIO)
        train_df = df.iloc[:train_end_idx]
        
        zero_variance_nodes = train_df.columns[train_df.std() == 0]
        if not zero_variance_nodes.empty:
            print(f"🛑 DIAGNOSTIC FAILED: The following nodes have zero variance in the training data:")
            print(f"   {zero_variance_nodes.tolist()}")
            print("   This will cause division by zero during standardization, leading to NaN values.")
            success = False

    except FileNotFoundError:
        print(f"🛑 DIAGNOSTIC FAILED: Processed discharge data not found at {config.PROCESSED_DISCHARGE_PATH}")
        success = False
    except Exception as e:
        print(f"🛑 DIAGNOSTIC FAILED: An unexpected error occurred while checking data: {e}")
        success = False
        
    if success:
        print("✅ Diagnostics complete. All checks passed.")
    print("---------------------------------\n")
    return success
    
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
    
def main():
    config = TrainingConfig()
    
    # if not pre_run_diagnostics(config):
    #     print("Aborting training due to diagnostic failures.")
    #     return
        
    print("--- Starting GraphWaveNet Training ---")
    train_loader, val_loader, _, adj_matrix = prepare_dataloaders(config=config)
    
    if train_loader is None:
        print("Failed to create dataloaders. Exiting.")
        return

    try:
        # Load the columns from a processed file which are the site IDs in order
        processed_df = pd.read_csv(config.PROCESSED_DATA_DIR / "processed_basin_chars.csv")
        site_order = processed_df['site_no'].astype(str).tolist()
    except FileNotFoundError:
        print("Warning: Could not load site order for plotting. Plots will use integer indices.")
        site_order = list(range(adj_matrix.shape[0]))


    supports = None
    print("USE ADJ TO FEED MODEL: ", config.USE_GRAPH)
    if config.USE_GRAPH:
        _, supports = get_adjacency_matrix_and_supports(adj_matrix, config.device)
    
    num_nodes = adj_matrix.shape[0]
    x_sample = next(iter(train_loader))[0]
    num_input_features = x_sample.shape[3]
    
    print(f"Device: {config.device}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of input features per node: {num_input_features}")
    
    model = gwnet(
        device=config.device,
        num_nodes=num_nodes,
        in_dim=num_input_features,
        out_dim=config.pred_len,
        supports=supports
    ).to(config.device)

    # load pretrained weights if available
    if Path(config.PRETRAIN_MODEL_PATH).exists():
        print(f"Loading pretrained model weights from {config.PRETRAIN_MODEL_PATH}")
        model.load_state_dict(torch.load(config.PRETRAIN_MODEL_PATH, map_location=config.device))

    
    # edge_index = torch.tensor(A.nonzero(), dtype=torch.long).contiguous().to(config.device) if A is not None else None
    # model = NhatModel(num_nodes=num_nodes, 
    #                       in_feat=num_input_features, 
    #                       hid_feat=32, 
    #                       num_blocks=3, 
    #                       T = config.seq_len, 
    #                       T_hat=config.pred_len,
    #                       edge_index=edge_index).to(config.device)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    try:
        with open(config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded node-wise scalers from {config.SCALER_PATH}")
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {config.SCALER_PATH}. Cannot perform evaluation correctly.")
        return
        
    trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, config, supports, scaler, site_order)
    trainer.train()
    
    print("--- Training Complete ---")
    print(f"Best model saved to {config.BEST_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()

    # config = TrainingConfig()
    # print("--- Starting GraphWaveNet Training ---")
    # train_loader, val_loader, _, adj_matrix = prepare_dataloaders(config=config)

    # try:
    #     with open(config.SCALER_PATH, 'rb') as f:
    #         scaler = pickle.load(f)
    #     print(f"Loaded node-wise scalers from {config.SCALER_PATH}")
    # except FileNotFoundError:
    #     print(f"Error: Scaler file not found at {config.SCALER_PATH}. Cannot perform evaluation correctly.")
    # all_y_true_unscaled = []
    # all_y_pred_unscaled = []
    # for x_batch, y_batch in tqdm(val_loader, desc="Validating", leave=False):
    #     y_batch = y_batch.to(config.device)
    #     y_target = y_batch.squeeze(-1)
    #     output = torch.zeros(y_target.shape)

    #     # unscale_data expects (batch, nodes, pred_len)
    #     y_target_unscaled = unscale_data(y_target, scaler[0])
    #     output_unscaled = y_target_unscaled[0,:,0].repeat(y_target_unscaled.shape[0], y_target_unscaled.shape[2], 1).permute(0,2,1)
    #     all_y_pred_unscaled.append(output_unscaled)
    #     all_y_true_unscaled.append(y_target_unscaled) # Keep y_target in (batch, nodes, pred_len) format

    # all_y_true = torch.cat(all_y_true_unscaled, dim=0)
    # all_y_pred = torch.cat(all_y_pred_unscaled, dim=0)

    # num_nodes = all_y_true.shape[1]
    # batch_size = all_y_true.shape[0]
    # nse_per_node = []
    # for i in range(num_nodes):
    #     for j in range(batch_size):
    #         y_true_node_i = all_y_true[j, i, :].flatten()
    #         y_pred_node_i = all_y_pred[j, i, :].flatten()
    #         node_nse = calculate_nse(y_true_node_i, y_pred_node_i)
    #         nse_per_node.append(node_nse)
    # valid_nses = [n for n in nse_per_node if abs(n) < 1e4]  # Filter out extreme NSE values
    # average_nse = np.mean(valid_nses) if valid_nses else -np.inf

    # print("AVERAG NSE: ", average_nse)