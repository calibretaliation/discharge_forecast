import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path
import os
import random
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import cKDTree # Added for proximity filtering

# from .data_processing import build_graph_and_adj_matrix, ensure_dir_exists, load_and_align_data, preprocess_static_features, preprocess_timeseries, run_data_processing

# Model/Training Config
INPUT_SEQ_LEN = 30
TARGET_SEQ_LEN = 10
NUM_TARGET_FEATURES = 1 # only predict discharge
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
BATCH_SIZE = 32

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SpatioTemporalDataset(Dataset):
    def __init__(self, X, Y, X_mask, Y_mask):
        """
        Args:
            X (np.array): Input features with shape 
                          (num_samples, input_seq_len, num_nodes, num_features_per_node)
            Y (np.array): Target sequences with shape 
                          (num_samples, target_seq_len, num_nodes, num_target_features)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.X_mask = torch.tensor(X_mask, dtype=torch.float32, requires_grad = False)
        self.Y_mask = torch.tensor(Y_mask, dtype=torch.float32, requires_grad = False)
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.X_mask[idx], self.Y_mask[idx] 
    
def scale_data_splits(train_data, val_data, test_data, scaler_save_path):
    """
    Fits scalers on training data for each node and feature individually.
    """
    _, num_nodes, num_features = train_data.shape
    print("SCALE SHAPE: ", train_data.shape)
    # all_scalers[0] will be a dict of scalers for discharge {node_idx: scaler}
    # all_scalers[1] will be a dict of scalers for rainfall {node_idx: scaler}
    all_scalers = [{} for _ in range(num_features)]

    scaled_train = np.copy(train_data)
    scaled_val = np.copy(val_data)
    scaled_test = np.copy(test_data)

    print(f"Fitting scalers on training data for {num_nodes} nodes and {num_features} features individually...")

    # Iterate over each feature 
    for feat_idx in range(num_features):
        if feat_idx == 0: # Feature 0 is discharge
            print("  Scaling discharge (feature 0) using StandardScaler for each node...")
            # scaler = StandardScaler()  # Use MinMaxScaler for discharge
        else: # Other features
            print(f"  Scaling rainfall/other (feature {feat_idx}) for each node...")
            
        for node_idx in range(num_nodes):
            scaler = StandardScaler()
            
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

def create_spatio_temporal_sequences(dynamic_data, static_features, all_mask,
                                     input_seq_len, target_seq_len, num_target_features=1):
    """Creates sliding window sequences for spatio-temporal forecasting."""
    num_timesteps, num_nodes, num_dynamic_feats = dynamic_data.shape
    if static_features is not None: 
        num_static_feats = static_features.shape[1] 
    else:
        num_static_feats = 0

    dynamic_data_reshaped = dynamic_data.reshape(num_timesteps, num_nodes, num_dynamic_feats)
    # print(dynamic_data_reshaped[:, -3, :])
    # print(all_mask[:, -3, :])
    # print(len(all_mask[:, -3, :]))
    num_total_input_features = num_dynamic_feats + num_static_feats
    
    X_list, Y_list = [], []
    X_list_mask, Y_list_mask = [], []
    for i in range(num_timesteps - input_seq_len - target_seq_len + 1):
        x_combined = dynamic_data_reshaped[i : i + input_seq_len, :, :]
        x_mask = all_mask[i : i + input_seq_len, :, :]
        static_features_tiled = np.tile(static_features, (input_seq_len, 1, 1))
        if static_features is not None:
            x_combined = np.concatenate((x_combined, static_features_tiled), axis=-1)
            x_mask = np.concatenate((x_mask, static_features_tiled), axis=-1)
        X_list.append(x_combined)
        X_list_mask.append(x_mask)
        
        y_slice = dynamic_data_reshaped[i + input_seq_len : i + input_seq_len + target_seq_len, :, :num_target_features]
        y_mask = all_mask[i + input_seq_len : i + input_seq_len + target_seq_len, :, :num_target_features]
        Y_list.append(y_slice)
        Y_list_mask.append(y_mask)
        # if y_slice[:, -3, :].sum() != 0:
            # print("Y SLICE FULL: ", y_slice[:, -3, :])
            # print("Y MASK FULL: ", y_mask[:,-3,:])
            # print("X SLICE FULL: ", x_combined[:, -3, :])
            # print("X MASK FULL: ", x_mask[:,-3,:])

    if not X_list:
        return np.array([]).reshape(0, input_seq_len, num_nodes, num_total_input_features), \
               np.array([]).reshape(0, target_seq_len, num_nodes, num_target_features)

    X = np.array(X_list)
    Y = np.array(Y_list)
    X_mask = np.array(X_list_mask)
    Y_mask = np.array(Y_list_mask)
    print(f"Created sequences: X shape {X.shape}, Y shape {Y.shape}")
    print(f"Created mask: X shape {X_mask.shape}, Y shape {Y_mask.shape}")
    return X, Y, X_mask, Y_mask


def ensure_dir_exists(directory_path):
    """Creates a directory if it doesn't already exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def filter_sites_by_proximity(source_sites_df, target_sites_df):
    """
    Finds the closest site in target_sites_df for each site in source_sites_df.
    
    Args:
        source_sites_df (pd.DataFrame): DataFrame with site locations to find matches for. 
                                        Must contain 'lat' and 'lon' columns.
        target_sites_df (pd.DataFrame): DataFrame of potential sites to be matched against.
                                        Must contain 'original_lat' and 'original_lon' columns.
    Returns:
        list: A unique list of site IDs from target_sites_df that are nearest neighbors.
    """
    print("--- Filtering CAMELS sites based on proximity to NJ sites ---")
    
    # Prepare source and target coordinates, dropping any missing values
    source_coords = source_sites_df[['original_lat', 'original_lon']].dropna()
    target_coords = target_sites_df[['original_lat', 'original_lon']].dropna()
    
    # Create a k-d tree for efficient spatial search on the target (CAMELS) sites
    tree = cKDTree(target_coords.values)
    
    # Query the tree to find the index of the single nearest neighbor for each source site
    _, indices = tree.query(source_coords.values, k=1)
    
    # Get the site IDs from the target dataframe using the found indices
    print(len(indices), "nearest neighbors found.")
    closest_sites = target_coords.index[indices].unique().tolist()
    
    print(f"Found {len(closest_sites)} unique CAMELS sites closest to the {len(source_sites_df)} NJ sites.")
    return closest_sites

def load_and_align_data(use_all_forcings = False, use_nj_sites = False, input_data_dir = "data_camels"):
    """
    Loads all necessary data, finds common sites, and aligns dataframes based on the configuration.
    """
    print("--- Loading and Aligning Data ---")
    
    DISCHARGE_CSV_PATH = f"{input_data_dir}/usgs_discharge.csv"
    BASIN_CHARS_CSV_PATH = f"{input_data_dir}/basin_characteristics.csv"
    FORCINGS_CSV_PATH = f"{input_data_dir}/camels_forcings.csv"
    RAINFALL_CSV_PATH = f"{input_data_dir}/rainfall_series.csv"
    NJ_BASIN_CHARS_CSV_PATH = "data/basin_characteristics.csv" 

    # Load core datasets
    discharge_df = pd.read_csv(DISCHARGE_CSV_PATH, dtype={'siteCode': str, 'value': float}, parse_dates=['dateTime'])
    basin_chars_df = pd.read_csv(BASIN_CHARS_CSV_PATH, dtype={'site_no': str}).set_index('site_no')
    
    # discharge_df.dropna(subset=['value'], inplace=True)  # Ensure no NaNs in critical columns
    # Pivot discharge data
    discharge_pivot = discharge_df.pivot_table(index='dateTime', columns='siteCode', values='value').sort_index()
    print(f"LOADED DISCHARGE PIVOT from {DISCHARGE_CSV_PATH} ", discharge_pivot.head())
    # Conditionally load forcing data
    if use_all_forcings:
        print("Configuration: Using all 5 forcing variables from camels_forcings.csv")
        forcing_df = pd.read_csv(FORCINGS_CSV_PATH, header=[0, 1], index_col=0, parse_dates=True)
        forcing_sites = set(forcing_df.columns.get_level_values(0))
    else:
        print("Configuration: Using only rainfall from rainfall_series.csv")
        forcing_df = pd.read_csv(RAINFALL_CSV_PATH, index_col=0, parse_dates=True)
        forcing_sites = set(forcing_df.columns)

    if use_nj_sites:
        print(f"Configuration: Filtering for sites listed in {NJ_BASIN_CHARS_CSV_PATH}")
        try:
            nj_sites_df = pd.read_csv(NJ_BASIN_CHARS_CSV_PATH, dtype={'site_no': str})
            sites_to_use = set(filter_sites_by_proximity(nj_sites_df, basin_chars_df))
        except FileNotFoundError:
            print(f"ERROR: New Jersey sites file not found at {NJ_BASIN_CHARS_CSV_PATH}. Aborting.")
            return None, None, None
        
        # Find the intersection of NJ sites and available CAMELS sites
        print(f"Found {len(sites_to_use)} sites in the main CAMELS attributes close to NJ site file.")
    else:
        print("Configuration: Using all available CAMELS sites.")
        sites_to_use = set(basin_chars_df.index)

    # Find common sites across all three datasets
    discharge_sites = set(discharge_pivot.columns)
    basin_char_sites = set(basin_chars_df.index)
    
    common_sites = sorted(list(discharge_sites.intersection(forcing_sites).intersection(basin_char_sites)))
    # common_sites = random.sample(common_sites, 109)
    print(f"Found {len(common_sites)} common sites across all required data files.")
    
    # Filter all dataframes to keep only common sites
    discharge_pivot = discharge_pivot[common_sites]
    basin_chars_df = basin_chars_df.loc[common_sites]

    if use_all_forcings:
        forcing_df = forcing_df[common_sites]
    else:
        forcing_df = forcing_df[common_sites]

    return discharge_pivot, forcing_df, basin_chars_df

def preprocess_static_features(basin_chars_df, config = None):
    """
    Cleans, scales, and prepares static basin characteristics.
    """
    print("\n--- Processing Static Basin Characteristics ---")
    # Attributes used in the GNN paper
    if config.DATASET == "camels":
        char_cols = [
            'p_mean', 
            'pet_mean', 'aridity', 'p_seasonality', 'frac_snow', 'high_prec_freq',
            'high_prec_dur', 'low_prec_freq', 'low_prec_dur', 'elev_mean', 'slope_mean',
            'area_gages2', 'frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff',
            'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity',
            'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac',
            'geol_permeability', 
            'carbonate_rocks_frac'
        ]
    else: 
        char_cols = [
            "CSL10_85","APRAVPRE","CONLENLDR","DRNAREA","FOREST","JUNAVPRE","PERMSSUR","POPDENS","SLOPLAGPER","STORAGE"
        ]
    
    
    # Filter for the specified attributes
    basin_features = basin_chars_df[char_cols].copy()
    
    # Clean data: drop any site with missing values in these critical attributes
    initial_sites = len(basin_features)
    basin_features.dropna(inplace=True)
    print(f"Dropped {initial_sites - len(basin_features)} sites due to missing static attributes.")
    
    # Scale each feature individually
    basin_features_scaled = basin_features.copy()

    # # DANGEROUS: RANDOMIZE static feature experiment
    # basin_features_scaled[:] = np.random.randint(0, 10, size=basin_features.shape)
    # print("DANGEROUS !!!!!")
    print(basin_features.head())
    print(basin_features_scaled.head())

    for col in basin_features_scaled.columns:
        scaler = StandardScaler()
        basin_features_scaled[col] = scaler.fit_transform(basin_features[[col]])
    
    print(f"Processed basin characteristics data shape: {basin_features_scaled.shape}")
    return basin_features_scaled

def preprocess_timeseries(df, log_transform=False):
    """Applies log1p transformation to a time series dataframe."""
    print(f"Timeseries NaN count: {df.isna().sum().sum()}")
    df = df.fillna(0) # Fill NaNs with a small value to avoid log(0)
    print(f"Timeseries NaN count after fill: {df.isna().sum().sum()}")

    if log_transform:
        log_const = - df.min().min()
        print(f"LOG CONST: {log_const}")
        df = np.log1p(df + log_const)
        print(f"Timeseries NaN count after log: {df.isna().sum().sum()}")
    return df

def build_graph_and_adj_matrix(static_features_df):
    """
    Builds a NetworkX graph and an adjacency matrix based on cosine similarity of static features.
    """
    print("\n--- Building Graph and Adjacency Matrix ---")
    sim_matrix = cosine_similarity(static_features_df.values)
    
    # Thresholding to create a sparse graph
    threshold = np.quantile(sim_matrix, 0.98)
    print(f"Using similarity threshold of {threshold:.4f} (75th percentile).")
    
    adj_matrix = (sim_matrix >= threshold).astype(int)
    np.fill_diagonal(adj_matrix, 1) # Add self-loops
    
    # Create the NetworkX graph object from the adjacency matrix
    graph = nx.from_numpy_array(adj_matrix)
    mapping = {i: site_code for i, site_code in enumerate(static_features_df.index)}
    nx.relabel_nodes(graph, mapping, copy=False)
    
    print(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    print(f"Built adjacency matrix with shape: {adj_matrix.shape}")
    return graph, adj_matrix

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

    ensure_dir_exists(config.PROCESSED_DATA_DIR)

    # Load and align data from all sources
    discharge_pivot, forcings_df, basin_chars_df = load_and_align_data(use_all_forcings = config.USE_ALL_FORCINGS, input_data_dir = config.INPUT_DATA_DIR)
    print("DISCHARGE PIVOT: ", discharge_pivot.shape)
    # Process static features to get the final list of valid sites
    static_features = preprocess_static_features(basin_chars_df, config)
    final_site_order = static_features.index.tolist()
    
    # Final alignment of time-series data based on valid static features
    print(f"\nPerforming final alignment to {len(final_site_order)} valid sites.")
    discharge_final = discharge_pivot[final_site_order]
    forcings_final = forcings_df[final_site_order]
    print("DISCHARGE : ", discharge_final[final_site_order[-3]])

    # Mask NaN before preprocessing for later use
    mask_discharge = ~discharge_final.isna()
    sample = random.randint(0, mask_discharge.shape[0])
    print(f"SAMPLE MASK : {mask_discharge.iloc[:,-3]}")
    print(f"SAMPLE data : {discharge_final.iloc[:,-3]}")
    mask_forcings = ~forcings_final.isna()
    all_mask = [mask_discharge]
    print("DISCHARGE MASK: ", mask_discharge.shape)
    # Preprocess the aligned time series
    print("\n--- Processing Time Series Data ---")
    df_describe = pd.DataFrame(forcings_final)
    print(df_describe.describe())
    
    processed_discharge = preprocess_timeseries(discharge_final, log_transform=config.LOG_TRANSFORM)
    processed_forcings = preprocess_timeseries(forcings_final, log_transform=config.LOG_TRANSFORM)
    
    log_const = None
    if config.LOG_TRANSFORM:
        log_const = -discharge_final.min().min()

    print("Log-transformed all time series.")
    # count NaN values
    print(f"Discharge NaN count: {processed_discharge.isna().sum().sum()}")
    print(f"Forcings NaN count: {processed_forcings.isna().sum().sum()}")
    # Stack all dynamic features for model input
    # Order is crucial: discharge must be the first feature (index 0)
    list_of_dynamic_features = [processed_discharge.values]
    
    if config.USE_ALL_FORCINGS:
        print("Stacking discharge and 5 forcing variables.")
        for var in ['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']:
            list_of_dynamic_features.append(processed_forcings.xs(var, axis=1, level=1).values)
            # all_mask.append()
    elif config.USE_DISCHARGE_ONLY:
        print("Using only discharge as dynamic feature.")
        # Only discharge is used, which is already in list_of_dynamic_features
        pass
    elif config.USE_RAINFALL:
        print("Stacking discharge and rainfall.")
        list_of_dynamic_features.append(processed_forcings.values)
        
    dynamic_df = np.stack(list_of_dynamic_features, axis=-1)
    mask_df = np.stack(all_mask, axis = -1)
    print("DYNAMIC DF: ", dynamic_df[:,-3,:])
    print("MASK DF: ", mask_df[:,-3,:])
    # Build the graph and adjacency matrix from the final scaled static features
    graph, adj_matrix = build_graph_and_adj_matrix(static_features)

    # _, dynamic_df, static_features, adj_matrix, _ = run_data_processing(save = False,
    #                                                                     use_all_forcings = config.USE_ALL_FORCINGS, 
    #                                                                     use_discharge_only = config.USE_DISCHARGE_ONLY, 
    #                                                                     use_rainfall = config.USE_RAINFALL,
    #                                                                     input_data_dir = config.INPUT_DATA_DIR, 
    #                                                                     processed_data_dir = config.PROCESSED_DATA_DIR)
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

    train_mask = mask_df[:train_end_idx]
    val_mask = mask_df[train_end_idx:val_end_idx]
    test_mask = mask_df[val_end_idx:]

    print("\n--- Checking for Distribution Shift ---")
    # Assuming feature 0 is discharge
    train_mean, train_std = np.mean(train_df[:, :, 0]), np.std(train_df[:, :, 0])
    val_mean, val_std = np.mean(val_df[:, :, 0]), np.std(val_df[:, :, 0])

    print(f"Train Set Stats (Discharge): Mean={train_mean:.2f}, Std={train_std:.2f}")
    print(f"Val Set Stats (Discharge):   Mean={val_mean:.2f}, Std={val_std:.2f}")

    print("Train Discharge Stats:")
    print(pd.DataFrame(train_df[:,:,0].flatten()).describe())

    print("\nValidation Discharge Stats:")
    print(pd.DataFrame(val_df[:,:,0].flatten()).describe())
    print("------------------------------------")

    print(f"Data split into Train ({len(train_df)}), Val ({len(val_df)}), Test ({len(test_df)}) sets.")

    # 3. Scale data splits (fit on train, transform all) and save scalers
    scaled_train_df, scaled_val_df, scaled_test_df = scale_data_splits(
        train_df, val_df, test_df, SCALER_SAVE_PATH
    )
    print("scaled train data: ", scaled_train_df[:,-3,:])

    print("\n--- Checking for Distribution Shift ---")
    # Assuming feature 0 is discharge
    train_mean, train_std = np.mean(scaled_train_df[:, :, 0]), np.std(scaled_train_df[:, :, 0])
    val_mean, val_std = np.mean(scaled_val_df[:, :, 0]), np.std(scaled_val_df[:, :, 0])

    print(f"Train Set Stats (Discharge): Mean={train_mean:.2f}, Std={train_std:.2f}")
    print(f"Val Set Stats (Discharge):   Mean={val_mean:.2f}, Std={val_std:.2f}")

    print("Train Discharge Stats:")
    print(pd.DataFrame(scaled_train_df[:,:,0].flatten()).describe())

    print("\nValidation Discharge Stats:")
    print(pd.DataFrame(scaled_val_df[:,:,0].flatten()).describe())
    print("------------------------------------")

    print(f"Scaled train data: {scaled_train_df.shape}")
    print(f"Train mask data: {train_mask.shape}")
    # exit()
    # 4. Create sequences from each scaled data split
    print("\n--- Creating Training Sequences ---")
    X_train, Y_train, X_train_mask, Y_train_mask = create_spatio_temporal_sequences(
        scaled_train_df, static_features, train_mask, input_seq_len, target_seq_len, num_target_features
    )
    # exit()

    print("\n--- Creating Validation Sequences ---")
    X_val, Y_val, X_val_mask, Y_val_mask = create_spatio_temporal_sequences(
        scaled_val_df, static_features, val_mask, input_seq_len, target_seq_len, num_target_features
    )
    print("\n--- Creating Test Sequences ---")
    X_test, Y_test, X_test_mask, Y_test_mask = create_spatio_temporal_sequences(
        scaled_test_df, static_features, test_mask, input_seq_len, target_seq_len, num_target_features
    )
    
    # 5. Create Datasets and DataLoaders
    train_dataset = SpatioTemporalDataset(X_train, Y_train, X_train_mask, Y_train_mask)
    val_dataset = SpatioTemporalDataset(X_val, Y_val, X_val_mask, Y_val_mask)
    test_dataset = SpatioTemporalDataset(X_test, Y_test, X_test_mask, Y_test_mask)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    print("\nDataLoaders created successfully.")
    
    return train_loader, val_loader, test_loader, adj_matrix, log_const


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
    pass