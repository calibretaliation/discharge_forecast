import torch
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# --- Import from our custom modules ---
from dataset.generate_training_data import prepare_dataloaders, get_adjacency_matrix_and_supports
from graph_wavenet import gwnet

# --- Configuration ---
# This class should mirror the one in train.py for consistency
class ShapConfig:
    # Data and Model Config
    seq_len = 30
    pred_len = 10
    
    # Paths
    PROCESSED_DATA_DIR = Path("dataset/processed_data_camels")
    ADJ_MATRIX_PATH = PROCESSED_DATA_DIR / "adjacency_matrix.pkl"
    STATIC_FEATURES_PATH = PROCESSED_DATA_DIR / "static_features_array.npy"
    MODEL_WEIGHTS_PATH = "weights/best_gwnet_model.pt"
    SHAP_PLOTS_DIR = Path("plots/shap_analysis")
    
    # SHAP explainer config
    NUM_BACKGROUND_SAMPLES = 50  # Number of training samples for background dataset
    NUM_EXPLAIN_SAMPLES = 100    # Number of test samples to explain
    
    # Training Config (needed for dataloader)
    batch_size = 32
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_feature_names(static_features_path):
    """
    Constructs the list of feature names for plotting.
    """
    # Dynamic features are known
    dynamic_features = [
        'Discharge', 'Precipitation', 'Solar Radiation', 'Max Temperature', 
        'Min Temperature', 'Vapor Pressure'
    ]
    
    # Load static feature names from the processed CSV header
    try:
        static_df = pd.read_csv("data/basin_characteristics.csv")
        # Use the same list of attributes as defined in the GNN paper
        static_cols_from_paper = [
            'p_mean', 'pet_mean', 'aridity', 'p_seasonality', 'frac_snow', 
            'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur', 
            'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max', 
            'lai_diff', 'gvf_max', 'gvf_diff', 'soil_depth_pelletier', 
            'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity', 
            'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 
            'geol_permeability', 'carbonate_rocks_frac'
        ]
        # Filter to ensure we only have columns that exist in the file
        static_features_names = [col for col in static_cols_from_paper if col in static_df.columns]
    except Exception as e:
        print(f"Warning: Could not load static feature names. Using generic names. Error: {e}")
        num_static = np.load(static_features_path).shape[1]
        static_features_names = [f'static_feat_{i}' for i in range(num_static)]
        
    return dynamic_features, static_features_names

# --- Define a PyTorch Module Wrapper for SHAP ---
# This class wraps the model and the necessary input permutation.
class GwnetModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GwnetModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        """
        Wrapper for the model's forward pass to handle permutation.
        Args:
            x (torch.Tensor): Input tensor of shape (N, seq_len, nodes, features)
        Returns:
            torch.Tensor: A 2D model output of shape (N, 1) for explanation.
        """
        # Permute the input tensor to match the model's expectation
        x_permuted = x.permute(0, 3, 2, 1) # (N, features, nodes, seq_len)
        output = self.model(x_permuted).squeeze(1) # (N, pred_len, nodes)
        
        # We explain the mean of the prediction for the first time step across all nodes
        mean_output = output[:, 0, :].mean(dim=1)
        
        # Ensure the output is 2D as expected by SHAP
        return mean_output.unsqueeze(1)

def main():
    """
    Main function to run the SHAP value calculation and plotting.
    """
    config = ShapConfig()
    config.SHAP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("--- Preparing Data Loaders ---")
    train_loader, _, test_loader, adj_matrix = prepare_dataloaders(config=config)
    
    if train_loader is None or test_loader is None:
        print("Failed to create dataloaders. Exiting.")
        return

    # --- Load Model and Data ---
    print("\n--- Loading Pre-trained Model and Data ---")
    
    # Get model parameters from a sample batch
    x_sample_batch, _ = next(iter(train_loader))
    num_nodes = x_sample_batch.shape[2]
    num_input_features = x_sample_batch.shape[3]
    
    # Load graph supports
    _, supports = get_adjacency_matrix_and_supports(config.ADJ_MATRIX_PATH, config.device)
    
    # Initialize model
    model = gwnet(
        device=config.device,
        num_nodes=num_nodes,
        in_dim=num_input_features,
        out_dim=config.pred_len,
        supports=supports
    ).to(config.device)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(config.MODEL_WEIGHTS_PATH, map_location=config.device))
        print(f"Successfully loaded model weights from {config.MODEL_WEIGHTS_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model weights not found at {config.MODEL_WEIGHTS_PATH}. Please train the model first.")
        return
        
    model.eval()

    # Create an instance of the model wrapper
    wrapped_model = GwnetModelWrapper(model)

    # --- Prepare Data for SHAP ---
    print("\n--- Preparing Background and Test Sets for SHAP ---")
    
    # Create a background dataset from the training loader
    background_data = []
    for x_batch, _ in train_loader:
        background_data.append(x_batch)
        if len(background_data) * config.batch_size >= config.NUM_BACKGROUND_SAMPLES:
            break
    background_data = torch.cat(background_data, dim=0)[:config.NUM_BACKGROUND_SAMPLES].to(config.device)

    # Create a test dataset to explain
    test_data = []
    for x_batch, _ in test_loader:
        test_data.append(x_batch)
        if len(test_data) * config.batch_size >= config.NUM_EXPLAIN_SAMPLES:
            break
    test_data = torch.cat(test_data, dim=0)[:config.NUM_EXPLAIN_SAMPLES].to(config.device)

    print(f"Background data shape: {background_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # --- Initialize SHAP Explainer ---
    print("\n--- Initializing SHAP DeepExplainer ---")
    explainer = shap.DeepExplainer(wrapped_model, background_data)

    # --- Calculate SHAP values ---
    print(f"\n--- Calculating SHAP values for {config.NUM_EXPLAIN_SAMPLES} test samples ---")
    shap_values = explainer.shap_values(test_data, check_additivity=False)
    
    # The output shape will be (num_samples, seq_len, num_nodes, num_features)
    print(f"SHAP values calculated with shape: {shap_values.shape}")

    # --- Visualize SHAP Values ---
    print("\n--- Generating SHAP Visualization Plots ---")
    
    # Reshape the SHAP values and test_data for plotting
    shap_values_reshaped = shap_values.reshape(-1, num_input_features)
    test_data_reshaped = test_data.cpu().numpy().reshape(-1, num_input_features)

    dynamic_feature_names, static_feature_names = get_feature_names(config.STATIC_FEATURES_PATH)
    all_feature_names = dynamic_feature_names + static_feature_names

    # Plot 1: Standard Summary Plot (dot version)
    shap.summary_plot(
        shap_values_reshaped,
        test_data_reshaped,
        feature_names=all_feature_names,
        show=False,
        plot_type='dot'
    )
    plt.gcf().set_size_inches(10, 12)
    plt.title("SHAP Feature Importance Summary (Dot Plot)")
    plt.tight_layout()
    save_path = config.SHAP_PLOTS_DIR / "feature_importance_summary_dot.png"
    plt.savefig(save_path, dpi=150)
    print(f"SHAP summary dot plot saved to: {save_path}")
    plt.close()

    # Plot 2: Bar Chart Summary Plot
    shap.summary_plot(
        shap_values_reshaped,
        test_data_reshaped,
        feature_names=all_feature_names,
        show=False,
        plot_type='bar'
    )
    plt.gcf().set_size_inches(10, 12)
    plt.title("Global Feature Importance (Mean Absolute SHAP Value)")
    plt.tight_layout()
    save_path = config.SHAP_PLOTS_DIR / "feature_importance_summary_bar.png"
    plt.savefig(save_path, dpi=150)
    print(f"SHAP summary bar plot saved to: {save_path}")
    plt.close()
    
    # Plot 3: Temporal Importance Plot
    # Average the absolute SHAP values across samples, nodes, and features for each time step
    # Shape of shap_values: (num_samples, seq_len, num_nodes, num_features)
    temporal_importance = np.abs(shap_values).mean(axis=(0, 2, 3))
    plt.figure(figsize=(12, 6))
    time_steps = np.arange(-config.seq_len + 1, 1)
    plt.plot(time_steps, temporal_importance, marker='o', linestyle='-')
    plt.xlabel("Time Step Relative to Prediction (t=0)")
    plt.ylabel("Mean Absolute SHAP Value")
    plt.title("Temporal Feature Importance")
    plt.grid(True)
    save_path = config.SHAP_PLOTS_DIR / "temporal_feature_importance.png"
    plt.savefig(save_path, dpi=150)
    print(f"Temporal importance plot saved to: {save_path}")
    plt.close()
    
    # Plot 4: Dynamic vs Static Feature Importance
    num_dynamic = len(dynamic_feature_names)
    mean_abs_shap_dynamic = np.abs(shap_values_reshaped[:, :num_dynamic]).mean()
    mean_abs_shap_static = np.abs(shap_values_reshaped[:, num_dynamic:]).mean()
    
    plt.figure(figsize=(8, 6))
    plt.bar(['Dynamic Features', 'Static Features'], [mean_abs_shap_dynamic, mean_abs_shap_static], color=['skyblue', 'salmon'])
    plt.ylabel("Overall Mean Absolute SHAP Value")
    plt.title("Aggregate Importance: Dynamic vs. Static Features")
    save_path = config.SHAP_PLOTS_DIR / "dynamic_vs_static_importance.png"
    plt.savefig(save_path, dpi=150)
    print(f"Dynamic vs. static importance plot saved to: {save_path}")
    plt.close()

    print("\n--- SHAP Analysis Complete ---")


if __name__ == "__main__":
    main()
