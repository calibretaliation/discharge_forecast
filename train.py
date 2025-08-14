# import sys, os
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from model import gwnet_new
from DSTAGNN import make_model
from ASTGCRN import ASTGCRN
# --- Configuration ---
class TrainingConfig:
    # Data and Model Config
    seq_len = 30                # Input sequence length (must match generate_training_data.py)
    pred_len = 10               # Target sequence length (must match generate_training_data.py)
    num_target_features = 1     # We predict only discharge
    in_feat = 1
    hid_feat = 64
    # Training Hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    batch_size = 32
    learning_rate = 1e-4
    
    MODEL = "gwnet"  # Model type, can be "gwnet" or "nhat"
    DATASET = "ours"
    if DATASET == "camels":
        # Path
        PROCESSED_DATA_DIR = Path("dataset/processed_masked_camels")
        # PRETRAIN_MODEL_PATH = "weights/camels/best_gwnet_model.pt"
        PRETRAIN_MODEL_PATH = "None" #Train from scratch
        BEST_MODEL_SAVE_PATH = f"weights/camels/{MODEL}_model_mask.pt"
        PLOT_SAVE_DIR = Path(f"plots/training/{MODEL}_masked_camels")
        INPUT_DATA_DIR = "dataset/data_camels"
    else:
        PROCESSED_DATA_DIR = Path("dataset/processed_masked")
        # PRETRAIN_MODEL_PATH = "weights/camels/best_gwnet_model.pt"
        PRETRAIN_MODEL_PATH = "None" #Train from scratch
        BEST_MODEL_SAVE_PATH = f"weights/ours/{MODEL}_model_no_mask.pt"
        PLOT_SAVE_DIR = Path(f"plots/training/{MODEL}_no_masked")
        INPUT_DATA_DIR = "dataset/data"
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
    USE_GRAPH = True
    LOG_TRANSFORM = True
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
    batch_size, pred_len, num_nodes = data.shape
    # Simpler loop for clarity in evaluation:
    for i in range(num_nodes):
        node_id = node_order[i]
        scaler = scaler_dict[node_id]
        # for j in range(batch_size):
        # Squeeze to make it (pred_len,) -> (pred_len, 1) for scaler
        node_ts = data[:, :, i] 
        unscaled_data[:, :, i] = scaler.inverse_transform(node_ts)

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
    def __init__(self, model, optimizer, scheduler, criterion, train_loader, val_loader, config, supports, scaler, site_order, log_const):
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
        self.log_const = log_const
        # self.scheduler = scheduler
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
        self.model.eval()
        with torch.no_grad():
            x_batch, y_batch, x_mask, y_mask = next(iter(self.val_loader))
            x_batch = x_batch.to(self.config.device)
            # x_mask = x_mask.to(self.config.device)
            
            x_permuted = x_batch.permute(0, 3, 2, 1)
            output = self.model(x_permuted).squeeze(1)
            row = 0
            col = 0
            nrows = 8
            ncols = 8
            fig, ax = plt.subplots(nrows= nrows, ncols = ncols, figsize = (40,20))
            while row < nrows:
                # Choose a random sample and node from the batch to plot
                sample_idx = np.random.randint(0, x_batch.size(0))
                node_idx = np.random.randint(0, x_batch.size(2))

                history_scaled = x_batch[sample_idx, :, node_idx, 0]  # Feature 0 is discharge
                truth_scaled = y_batch[sample_idx, :, node_idx, 0] 
                pred_scaled = output[sample_idx, :, node_idx]
                # Unscale for plotting
                discharge_scaler_dict = self.scaler
                # print(discharge_scaler_dict)
                truth_unscaled = unscale_data(truth_scaled.unsqueeze(0).unsqueeze(0), discharge_scaler_dict)[0,0,:]
                pred_unscaled = unscale_data(pred_scaled.unsqueeze(0).unsqueeze(0), discharge_scaler_dict)[0,0,:]
                history_unscaled = unscale_data(history_scaled.unsqueeze(0).unsqueeze(0), discharge_scaler_dict)[0,0,:]

                # Reverse the log transform
                if self.config.LOG_TRANSFORM:
                    truth_unscaled = torch.expm1(truth_unscaled) - self.log_const
                    pred_unscaled = torch.expm1(pred_unscaled) - self.log_const
                    history_unscaled = torch.expm1(history_unscaled) - self.log_const
                
                # truth_unscaled = truth_unscaled * y_mask[sample_idx, :, node_idx, 0]
                # history_unscaled = history_unscaled * x_mask[sample_idx, :, node_idx, 0]
                total_len = len(history_unscaled) + len(truth_unscaled)
                # Plot history
                ax[row,col].plot(np.arange(0, len(history_unscaled)), history_unscaled.cpu(), label='Input History', color='gray')
                # Plot ground truth
                ax[row,col].plot(np.arange(len(history_unscaled), total_len), truth_unscaled.cpu(), label='Ground Truth', color='blue', marker='o')
                # Plot prediction
                ax[row,col].plot(np.arange(len(history_unscaled), total_len), pred_unscaled.cpu(), label='Prediction', color='red', linestyle='--')

                ax[row,col].set_xlabel('Time Steps')
                ax[row,col].set_ylabel('Discharge')

                col += 1
                if col >= ncols:
                    col = 0
                    row += 1
                
            fig.suptitle(f'Sample Prediction vs. Ground Truth (Epoch {epoch})')
            fig.legend()
            save_path = self.config.PLOT_SAVE_DIR / f"prediction_epoch_{epoch}.png"
            plt.savefig(save_path)
            print(f"Plotting sample prediction to {save_path}")
            plt.close()

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        progress = tqdm(self.train_loader, desc="Training", leave=False)
        for x_batch, y_batch, x_mask, y_mask in progress:
            x_batch = x_batch.to(self.config.device)
            y_batch = y_batch.to(self.config.device)
            x_mask = x_mask.to(self.config.device)
            y_mask = y_mask.to(self.config.device)
            self.optimizer.zero_grad()
            
            # Dataloader provides: (batch, seq_len, nodes, features)
            # Model's Conv2d expects: (batch, features, nodes, seq_len)
            x_batch = x_batch * x_mask
            x_permuted = x_batch.permute(0, 3, 2, 1)
            output = self.model(x_permuted) # batch, pred_len, num_nodes
            y_target = y_batch.squeeze(-1) # -> (batch_size, pred_len, num_nodes)
            y_mask_bool = y_mask.squeeze(-1).bool()
            output = output
            y_target = y_target
            loss = self.criterion(output, y_target)

            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            train_loss = total_loss / len(self.train_loader) + 0.3
            progress.set_postfix({'loss': f'{(total_loss/(progress.n + 1)):.2f}'})
        return train_loss
        
    
    def eval_epoch(self):
        self.model.eval()
        total_loss = 0
        all_y_true_unscaled = []
        all_y_pred_unscaled = []
        all_y_mask = []
        # print(self.scaler)
        with torch.no_grad():
            for x_batch, y_batch, x_mask, y_mask in tqdm(self.val_loader, desc="Validating", leave=False):
                x_batch = x_batch.to(self.config.device)
                y_batch = y_batch.to(self.config.device)
                x_mask = x_mask.to(self.config.device)
                y_mask = y_mask.to(self.config.device)
                
                # self.optimizer.zero_grad()
                x_batch = x_batch * x_mask
                x_permuted = x_batch.permute(0, 3, 2, 1)
                output = self.model(x_permuted)
                # .squeeze(1)  # -> (batch_size, pred_len, num_nodes)
                y_target = y_batch.squeeze(-1)  # -> (batch_size, pred_len, num_nodes)
                y_mask_bool = y_mask.squeeze(-1).bool()
                # print(f"Y MASK BOOL: {y_mask_bool}\nShape: {y_mask_bool.shape}")
                # print(f"Y TARGET: {y_target}\nShape: {y_target.shape}")
                output_masked = output
                y_target_masked = y_target

                loss = self.criterion(output_masked, y_target_masked)
                if not torch.isnan(loss):
                    total_loss += loss.item()

                output_unscaled = unscale_data(output, self.scaler)
                y_target_unscaled = unscale_data(y_target, self.scaler)
                if self.config.LOG_TRANSFORM:
                    y_target_unscaled = torch.expm1(y_target_unscaled) - self.log_const
                    output_unscaled = torch.expm1(output_unscaled) - self.log_const
                # y_target_unscaled = y_target_unscaled * y_mask.squeeze(-1).cpu()
                all_y_pred_unscaled.append(output_unscaled)
                all_y_true_unscaled.append(y_target_unscaled)
                all_y_mask.append(y_mask_bool) # Append the mask for NSE calculation

        all_y_true = torch.cat(all_y_true_unscaled, dim=0)
        all_y_pred = torch.cat(all_y_pred_unscaled, dim=0)
        all_y_mask = torch.cat(all_y_mask, dim=0)
        print(all_y_true.shape, all_y_pred.shape, all_y_mask.shape)
        num_nodes = all_y_true.shape[2]
        batch_size = all_y_true.shape[0]
        valid_nses = []
        print(f"CALCULATING NSE FOR {num_nodes} NODES AND {batch_size} BATCH SIZE")
        for i in range(num_nodes):
            nse_per_node = []
            # for j in range(batch_size):
            y_mask_i = all_y_mask[:, :, i].flatten().cpu()
            y_true_node_i = all_y_true[:, :, i].flatten()
            y_pred_node_i = all_y_pred[:, :, i].flatten()
            if len(y_pred_node_i) == 0:
                print(f"Y MASK: {y_mask_i.shape}")
                print(f"Y TRUE: {y_true_node_i.shape}")
            
            node_nse = calculate_nse(y_true_node_i, y_pred_node_i)
            nse_per_node.append(node_nse)

            valid_nses.append(np.mean(nse_per_node))
        valid_nses = [n for n in valid_nses if abs(n) < 1e8]  # Filter out extreme NSE values
        average_nse = np.mean(valid_nses) if valid_nses else -np.inf
        # print(valid_nses)
        avg_loss = total_loss / len(self.val_loader) + 0.3
        return avg_loss, average_nse, valid_nses

    def train(self):
        self.config.PLOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        Path(self.config.BEST_MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
        best_val_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            avg_train_loss = self.train_epoch()
            avg_val_loss, val_nse, valid_nses = self.eval_epoch()
            # self.scheduler.step(avg_val_loss)
            self.train_loss_history.append(avg_train_loss)
            self.val_loss_history.append(avg_val_loss)
            self.val_nse_history.append(val_nse)

            print(f"Epoch {epoch}/{self.config.num_epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val NSE: {val_nse:.4f}")

            if avg_val_loss < best_val_loss and not np.isnan(avg_val_loss):
                best_val_loss = avg_val_loss
                print(f"New best validation loss: {best_val_loss:.4f}. Saving model to {self.config.BEST_MODEL_SAVE_PATH}")
                torch.save(self.model.state_dict(), self.config.BEST_MODEL_SAVE_PATH)
                try:
                    new_adj = self.model.get_learned_adj()[0]
                except Exception as e:
                    print(f"Error getting learned adjacency matrix: {e}")
                    new_adj = None
                if new_adj is not None:
                    # print("Learned adjacency matrix shape:", new_adj[:10,:10])
                    # Save the learned adjacency matrix for inspection
                    with open(self.config.NEW_ADJ_MATRIX_PATH, 'wb') as f:
                        pickle.dump(new_adj.cpu().detach().numpy(), f)
                    # print(f"Saved learned adjacency matrix to {self.config.NEW_ADJ_MATRIX_PATH}")
                self.plot_sample_prediction(epoch)
                print("TOTAL NSE: ", len(valid_nses))
                with open(self.config.PROCESSED_DATA_DIR / "valid_nses.txt", 'w') as f:
                    for nse in valid_nses:
                        f.write(f"{nse}\n")
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
    train_loader, val_loader, _, adj_matrix, log_const = prepare_dataloaders(config=config)

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
    
    if config.MODEL == "gwnet":
        model = gwnet(
            device=config.device,
            num_nodes=num_nodes,
            in_dim=num_input_features,
            out_dim=config.pred_len,
            supports=supports
        ).to(config.device)
    elif config.MODEL == "nhat":    
        model = gwnet_new(
            device=config.device,
            num_nodes=num_nodes,
            in_dim=num_input_features,
            out_dim=config.pred_len,
            supports=supports
        ).to(config.device)
    elif config.MODEL == "dstagnn":
        adj_matrix = np.load("/aul/homes/nhoan009/discharge_forecast/dataset/data_camels/dstagnn_adjacency_matrix.npy")
        model = make_model(config.device, config.in_feat, nb_block = 2, in_channels = config.in_feat, K = 2, nb_chev_filter = 16, nb_time_filter = 16, 
                           time_strides = 1, adj_mx = adj_matrix, adj_pa = adj_matrix, adj_TMD = adj_matrix, num_for_predict = config.pred_len, len_input = config.seq_len, num_of_vertices = num_nodes, 
                           d_model = 64, d_k = 16, d_v = 16, n_heads = 4)
    elif config.MODEL == "astgcrn":
        model = ASTGCRN(
            num_nodes=num_nodes,
            input_dim=num_input_features,
            hidden_dim = 64,
            output_dim = 1,
            seq_len = config.seq_len,
            horizon = config.pred_len,
            num_layers = 3,
            device = config.device).to(config.device)
    # load pretrained weights if available
    # if Path(config.PRETRAIN_MODEL_PATH).exists():
    #     print(f"Loading pretrained model weights from {config.PRETRAIN_MODEL_PATH}")
    #     model.load_state_dict(torch.load(config.PRETRAIN_MODEL_PATH, map_location=config.device))
    
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, "min")

    try:
        with open(config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded node-wise scalers from {config.SCALER_PATH}")
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {config.SCALER_PATH}. Cannot perform evaluation correctly.")
        return
    trainer = Trainer(model, optimizer, scheduler, criterion, train_loader, val_loader, config, supports, scaler, site_order, log_const)
    trainer.train()
    
    print("--- Training Complete ---")
    print(f"Best model saved to {config.BEST_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
