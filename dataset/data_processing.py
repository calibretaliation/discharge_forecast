import random
import pandas as pd
import numpy as np
import pickle
import os
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import cKDTree # Added for proximity filtering


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

def preprocess_static_features(basin_chars_df):
    """
    Cleans, scales, and prepares static basin characteristics.
    """
    print("\n--- Processing Static Basin Characteristics ---")
    # Attributes used in the GNN paper
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
    
    # char_cols = [
    #     "CSL10_85","APRAVPRE","CONLENLDR","DRNAREA","FOREST","JUNAVPRE","PERMSSUR","POPDENS","SLOPLAGPER","STORAGE"
    # ]
    
    
    # Filter for the specified attributes
    basin_features = basin_chars_df[char_cols].copy()
    
    # Clean data: drop any site with missing values in these critical attributes
    initial_sites = len(basin_features)
    basin_features.dropna(inplace=True)
    print(f"Dropped {initial_sites - len(basin_features)} sites due to missing static attributes.")
    
    # Scale each feature individually
    basin_features_scaled = basin_features.copy()

    # DANGEROUS: RANDOMIZE static feature experiment
    basin_features_scaled[:] = np.random.randint(0, 10, size=basin_features.shape)
    print("DANGEROUS !!!!!")
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
        df = np.log1p(df)
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

def run_data_processing(save = True, use_all_forcings = False, use_discharge_only = True, use_rainfall = False, input_data_dir = "data_camels", processed_data_dir = "processed_data_camels"):
    """Executes the full data processing pipeline."""
    ensure_dir_exists(processed_data_dir)

    # 1. Load and align data from all sources
    discharge_pivot, forcings_df, basin_chars_df = load_and_align_data(use_all_forcings = use_all_forcings, input_data_dir = input_data_dir)

    # 2. Process static features to get the final list of valid sites
    static_features_scaled_df = preprocess_static_features(basin_chars_df)
    final_site_order = static_features_scaled_df.index.tolist()
    
    # 3. Final alignment of time-series data based on valid static features
    print(f"\nPerforming final alignment to {len(final_site_order)} valid sites.")
    discharge_final = discharge_pivot[final_site_order]
    forcings_final = forcings_df[final_site_order]
    
    # 4. Preprocess (log-transform) the aligned time series
    print("\n--- Processing Time Series Data ---")
    df_describe = pd.DataFrame(forcings_final)
    print(df_describe.describe())
    processed_discharge = preprocess_timeseries(discharge_final, log_transform=False)
    processed_forcings = preprocess_timeseries(forcings_final, log_transform=False)
    print("Log-transformed all time series.")
    # count NaN values
    print(f"Discharge NaN count: {processed_discharge.isna().sum().sum()}")
    print(f"Forcings NaN count: {processed_forcings.isna().sum().sum()}")
    # 5. Stack all dynamic features for model input
    # Order is crucial: discharge must be the first feature (index 0)
    list_of_dynamic_features = [processed_discharge.values]
    
    if use_all_forcings:
        print("Stacking discharge and 5 forcing variables.")
        for var in ['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']:
            list_of_dynamic_features.append(processed_forcings.xs(var, axis=1, level=1).values)
    elif use_discharge_only:
        print("Using only discharge as dynamic feature.")
        # Only discharge is used, which is already in list_of_dynamic_features
        pass
    elif use_rainfall:
        print("Stacking discharge and rainfall.")
        list_of_dynamic_features.append(processed_forcings.values)
        
    dynamic_features_array = np.stack(list_of_dynamic_features, axis=-1)
    
    # 6. Build the graph and adjacency matrix from the final scaled static features
    graph, adjacency_matrix = build_graph_and_adj_matrix(static_features_scaled_df)

    if save:
        ADJ_MATRIX_OUTPUT_PATH = os.path.join(processed_data_dir, "adjacency_matrix.pkl")
        GRAPH_OUTPUT_PATH = os.path.join(processed_data_dir, "usgs_graph.gpickle") # Path for the graph object
        STATIC_FEATURES_OUTPUT_PATH = os.path.join(processed_data_dir, "static_features_array.npy")
        DYNAMIC_FEATURES_OUTPUT_PATH = os.path.join(processed_data_dir, "dynamic_features.npy")


        # 7. Save all processed artifacts
        processed_discharge.to_csv(os.path.join(processed_data_dir, "processed_discharge.csv"), index=True)

        print("\n--- Saving Processed Artifacts ---")
        np.save(DYNAMIC_FEATURES_OUTPUT_PATH, dynamic_features_array)
        print(f"Saved combined dynamic features to: {DYNAMIC_FEATURES_OUTPUT_PATH} (Shape: {dynamic_features_array.shape})")
        
        np.save(STATIC_FEATURES_OUTPUT_PATH, static_features_scaled_df.values)
        print(f"Saved static features array to: {STATIC_FEATURES_OUTPUT_PATH} (Shape: {static_features_scaled_df.shape})")

        with open(ADJ_MATRIX_OUTPUT_PATH, "wb") as f:
            pickle.dump(adjacency_matrix, f)
        print(f"Saved adjacency matrix to: {ADJ_MATRIX_OUTPUT_PATH}")
        
        with open(GRAPH_OUTPUT_PATH, "wb") as f:
            pickle.dump(graph, f)
        print(f"Saved graph object to: {GRAPH_OUTPUT_PATH}")

        print("\n--- Data processing pipeline complete. ---")
    else: 
        return processed_discharge, dynamic_features_array, static_features_scaled_df, adjacency_matrix, graph
if __name__ == "__main__":

    run_data_processing()
