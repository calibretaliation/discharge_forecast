from scipy.optimize import linprog
import time
import pandas as pd
import numpy as np
from pathlib import Path
import os
import random
from tqdm import tqdm
def wasserstein_distance(p, q, D):
    """
    Calculates the Wasserstein distance (Earth Mover's Distance) between
    two probability distributions p and q, given a cost matrix D.
    This is solved as a linear programming problem.
    """
    # Create the equality constraint matrix A_eq
    A_eq = []
    # Constraint for p: sum of transported mass from each source i must equal p[i]
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.flatten())
    # Constraint for q: sum of transported mass to each destination j must equal q[j]
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.flatten())

    A_eq = np.array(A_eq)
    # Combine the probability distributions for the equality constraints
    b_eq = np.concatenate([p, q])
    
    # Flatten the cost matrix for the linear programming solver
    D_flat = D.flatten()

    # Solve the linear programming problem
    # We can drop one constraint as the total mass is conserved (sum(p) == sum(q) == 1)
    # start = time.time()
    result = linprog(c=D_flat, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    # print("lin_prog time: ", time.time() - start)
    
    return result.fun if result.success else float('inf')

def spatial_temporal_aware_distance(x, y):
    """
    Calculates the Spatial-Temporal Aware Distance (STAD) between two
    time-series x and y.
    """
    # start = time.time()
    # Ensure inputs are numpy arrays
    x, y = np.array(x), np.array(y)

    # Step 1: Create probability distributions (p, q)
    # Calculate the L2 norm for each day's vector to represent daily volume
    x_norm = (x**2).sum(axis=1, keepdims=True)**0.5
    y_norm = (y**2).sum(axis=1, keepdims=True)**0.5
    
    # print("x_norm time: ", time.time() - start)
    # start = time.time()
    x_norm = np.nan_to_num(x_norm, nan=0.0)
    y_norm = np.nan_to_num(y_norm, nan=0.0)
    # print("fill nan time: ", time.time() - start)
    # start = time.time()

    # Handle potential division by zero if a day has zero flow
    x_norm[x_norm == 0] = 1e-4
    y_norm[y_norm == 0] = 1e-4

    # Normalize daily volumes to get probability distributions
    p = x_norm.flatten() / x_norm.sum()
    q = y_norm.flatten() / y_norm.sum()

    # Step 2: Create the cost matrix (D) using cosine distance
    # Normalize each daily vector before calculating cosine similarity
    x_normalized_daily = x / x_norm
    y_normalized_daily = y / y_norm
    D = 1 - np.dot(x_normalized_daily, y_normalized_daily.T)
    # check for nan and inf in D
    if np.isnan(D).any() or np.isinf(D).any():
        # print("Warning: Cost matrix D contains NaN or Inf values. Replacing with zeros.")
        D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    # Step 3: Compute Wasserstein Distance
    return wasserstein_distance(p, q, D)
BASIN_CHARS_PATH = Path("data_camels/basin_characteristics.csv") # Used for getting site IDs
all_sites = pd.read_csv(BASIN_CHARS_PATH, dtype={'site_no': str})['site_no'].astype(str).tolist()
DATA_DIR = Path("./data_camels")
DISCHARGE_PATH = DATA_DIR / "usgs_discharge.csv"

discharge_df = pd.read_csv(DISCHARGE_PATH, parse_dates=['dateTime'], dtype={'siteCode': str})
discharge_pivot = discharge_df.pivot(index='dateTime', columns='siteCode', values='value')
discharge_pivot = discharge_pivot[all_sites]  # Ensure we only use the sites we have characteristics for

TIMESTEPS_PER_SEQUENCE = 100 # This is 'dr' from the paper's methodology
SPARSITY = 0.05             # Keep edges for the top 5% most similar nodes for each node

# Get data dimensions
total_timesteps, num_nodes = discharge_pivot.shape
num_sequences = total_timesteps // TIMESTEPS_PER_SEQUENCE
truncated_timesteps = num_sequences * TIMESTEPS_PER_SEQUENCE
data_truncated = discharge_pivot.iloc[:truncated_timesteps].values

data_reshaped = data_truncated.reshape(num_sequences, TIMESTEPS_PER_SEQUENCE, num_nodes)
print(f"Original data shape: {discharge_pivot.shape}")
print(f"Reshaped data for STAD calculation: {data_reshaped.shape}")
print(f"-> {data_reshaped.shape[0]} sequences of {data_reshaped.shape[1]} timesteps for {data_reshaped.shape[2]} nodes.")
similarity_matrix = np.zeros((num_nodes, num_nodes))
print(f"Calculating STAD similarity matrix for {num_nodes} nodes...")
start_time = time.time()

for i in tqdm(range(num_nodes), leave = False, position = 0):
    print(f"\rProcessing node {i + 1}/{num_nodes}", flush=True)
    for j in tqdm(range(i, num_nodes), leave = False, position = 1): # Use symmetry to reduce calculations
        if i == j:
            similarity_matrix[i, j] = 1.0
            continue
        
        # Extract data for node i and node j
        node_i_data = data_reshaped[:, :, i]
        node_j_data = data_reshaped[:, :, j]
        
        # Calculate STAD and convert to similarity
        stad = spatial_temporal_aware_distance(node_i_data, node_j_data)
        similarity = 1 - stad
        
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity # Symmetric matrix
    

end_time = time.time()
print(f"\nSimilarity matrix calculation finished in {end_time - start_time:.2f} seconds.")
print("-" * 50)

adj_matrix = np.zeros((num_nodes, num_nodes))
top_k = int(num_nodes * SPARSITY)

print(f"Generating sparse adjacency matrix with sparsity={SPARSITY} (top {top_k} connections per node)...")

for i in range(num_nodes):
    # Get the indices of the top_k most similar nodes for node i
    # We use .argsort() which sorts in ascending order, so we take from the end
    top_k_indices = similarity_matrix[i, :].argsort()[-top_k:]
    
    # Set the corresponding entries in the adjacency matrix to 1
    adj_matrix[i, top_k_indices] = 1

print("Adjacency matrix generated.")
print(f"Final Adjacency Matrix Shape: {adj_matrix.shape}")
print(f"Sparsity of adjacency matrix: {adj_matrix.sum() / (num_nodes * num_nodes):.4f}")

# Display the first 5x5 part of the matrix as an example
print("\nAdjacency Matrix (first 5x5):")
print(adj_matrix[:5, :5])

# save adj_matrix and similarity matrix to file
np.save(DATA_DIR / "dstagnn_adjacency_matrix.npy", adj_matrix)
np.save(DATA_DIR / "dstagnn_similarity_matrix.npy", similarity_matrix)