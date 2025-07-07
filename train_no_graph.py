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
from dataset.generate_training_data import prepare_dataloaders, get_adjacency_matrix_and_supports

class LSTMModel(nn.Module):
    # a LSTM model takes in a sequence of time series data and predicts the next 10 values
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        # convert from (B, T, N, C) to (B*N, T, C)
        B, T, N, C = x.size()
        x = x.permute(0, 1, 3, 2).contiguous().view(x.size(0) * x.size(2), x.size(1), x.size(3))  # (batch_size * num_nodes, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        last_time_step = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        output = self.fc(last_time_step)  # (batch_size, output_size)
        #convert back to (B, N, T)
        output = output.view(B, N, -1).permute(0,2,1)  # (batch_size, num_nodes, output_size)
        return output

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
    SCALER_PATH = PROCESSED_DATA_DIR / "timeseries_node_scalers.pkl"
    ADJ_MATRIX_PATH = PROCESSED_DATA_DIR / "adjacency_matrix.pkl"
    PROCESSED_DISCHARGE_PATH = PROCESSED_DATA_DIR / "processed_discharge.csv"
    BEST_MODEL_SAVE_PATH = "weights/best_gwnet_model.pt"
    PLOT_SAVE_DIR = Path("plots/training/camels_discharge_only")
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15

    USE_RAINFALL = False         # Set to False to use only 1 time series (discharge)
    USE_STATIC_FEATURES = False   # Set to False to exclude static basin attributes
    USE_ALL_FORCINGS = False
    USE_DISCHARGE_ONLY = True
    INPUT_DATA_DIR = "dataset/data_camels"

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
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, config, scaler, site_order):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
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
            
            output = self.model(x_batch)     
            # Target from dataloader is (batch, pred_len, nodes, 1)
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
            # print("node sample: ", x_permuted[0,:,random.randint(0, x_permuted.shape[2] - 1),0])
            output = self.model(x_batch)     
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
                output = self.model(x_batch)     
                # Target from dataloader is (batch, pred_len, nodes, 1)
                # Squeeze the last dimension to match model output
                y_target = y_batch.squeeze(-1) # -> (batch_size, pred_len, num_nodes)

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
                # self.plot_sample_prediction(epoch)
            self.plot_metrics()

def main():
    config = TrainingConfig()
    
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
    
    num_nodes = adj_matrix.shape[0]
    x_sample = next(iter(train_loader))[0]
    num_input_features = x_sample.shape[3]
    
    print(f"Device: {config.device}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of input features per node: {num_input_features}")
    
    model = LSTMModel(
        input_size=num_input_features,
        hidden_size=64,  
        num_layers=3,    # Number of LSTM layers
        output_size=config.pred_len  
    ).to(config.device)

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    try:
        with open(config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded node-wise scalers from {config.SCALER_PATH}")
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {config.SCALER_PATH}. Cannot perform evaluation correctly.")
        return
        
    trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, config, scaler, site_order)
    trainer.train()
    
    print("--- Training Complete ---")
    print(f"Best model saved to {config.BEST_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
