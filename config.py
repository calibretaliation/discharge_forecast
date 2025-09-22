import torch
from pathlib import Path

class Config:
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

