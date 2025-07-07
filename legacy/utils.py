from pathlib import Path
import random
import numpy as np
import torch
from dateutil import parser
import matplotlib.pyplot as plt


class NodeStandardScaler:
    """
    Z-score normalise *independently* for every node, i.e.
        X_scaled[n, t] = (X[n, t] − μ_n) / σ_n
    with μ_n and σ_n computed over the full time axis.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps   = eps
        self.mu    = None   # shape (N, 1)
        self.std   = None   # shape (N, 1)
        self.fitted = False

    # --------------------------------------------------------------------- fit
    def fit(self, data: torch.Tensor) -> "NodeStandardScaler":
        """
        data : tensor (N_nodes, T_total)
        """
        if data.dim() != 2:
            raise ValueError("Expect data of shape (N_nodes, T_total)")

        self.mu  = data.mean(dim=1, keepdim=True)                       # (N, 1)
        self.std = data.std(dim=1, keepdim=True, unbiased=False)        # (N, 1)
        self.std = self.std.clamp_min(self.eps)                         # avoid /0
        self.fitted = True
        return self

    # -------------------------------------------------- utility
    def _to(self, device: torch.device):
        """Internal helper: copy μ and σ to *device* and return them."""
        return self.mu.to(device), self.std.to(device)

    # -------------------------------------------------- transform
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet.")
        mu, std = self._to(x.device)
        return (x - mu) / std

    def inverse_transform(self, x_scaled: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("Scaler has not been fitted yet.")
        mu, std = self._to(x_scaled.device)
        return x_scaled * std + mu

    # -------------------------------------------------------------------- I/O
    def save(self, file_path: str | Path) -> None:
        if not self.fitted:
            raise RuntimeError("Nothing to save – fit the scaler first.")
        torch.save({"mu": self.mu, "std": self.std}, file_path)
    @classmethod
    def load(cls, file_path: str | Path, eps: float = 1e-8) -> "NodeStandardScaler":
        obj = cls(eps)
        ckpt = torch.load(file_path, map_location="cpu")
        obj.mu  = ckpt["mu"]
        obj.std = ckpt["std"]
        obj.fitted = True
        return obj

class NodeStandardScaler3D:
    """
    Per-(node, feature) standardisation for tensors shaped (N, F, T).

      x_scaled = (x - mu) / std
      x        =  x_scaled * std + mu
    """
    def fit(self, x: torch.Tensor):
        # x  : [N, F, T]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        self.mu  = x.mean(dim=2, keepdim=True)                # [N, F, 1]
        self.std = x.std (dim=2, keepdim=True).clamp_min(1e-6)
        return self

    def _to(self, device: torch.device):
        """Internal helper: copy μ and σ to *device* and return them."""
        return self.mu.to(device), self.std.to(device)
    
    def transform(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)

        mu, std = self._to(x.device)
        return (x - mu) / std

    def inverse_transform(self, x_scaled: torch.Tensor):
        if not isinstance(x_scaled, torch.Tensor):
            x_scaled = torch.as_tensor(x_scaled, dtype=torch.float32)

        mu, std = self._to(x_scaled.device)
        return x_scaled * std + mu

    def save(self, path: str = "node_scaler.pt"):
        torch.save({"mu": self.mu, "std": self.std}, path)

    @staticmethod
    def load(path: str = "node_scaler.pt"):
        ckpt = torch.load(path, map_location="cpu")
        sc   = NodeStandardScaler3D()
        sc.mu  = ckpt["mu"]
        sc.std = ckpt["std"]
        return sc
    
def inverse_one_feature(x_scaled, scaler, feat_idx=0):
    """
    x_scaled : (B, N, P)   –  scaled prediction of ONE variable
    returns   : (B, N, P)   –  back-to-raw units
    """
    B, N, P = x_scaled.shape
    mu  = scaler.mu[:, feat_idx, 0].to(x_scaled.device)   # [N]
    std = scaler.std[:, feat_idx, 0].to(x_scaled.device)  # [N]

    # reshape for broadcasting: [B, N, P] * [N,1] + [N,1]
    mu  = mu.view(1, N, 1)
    std = std.view(1, N, 1)
    return x_scaled * std + mu

def parse_date(date_str):
    """
    Parse an ISO date string (e.g., '2024-01-01T00:00:00.000') into a datetime object.
    """
    return parser.parse(date_str)

def extract_series_from_node(node_data, start_date, end_date, config):
    """
    Given node data, filter the series to the given [start_date, end_date] and return:
      - A list of datetime objects (sorted) as the reference time index.
      - A NumPy array of the corresponding values.
    """
    series_type=config.series_type
    series = node_data.get(series_type, [])
    filtered = []
    for record in series:
        try:
            dt = parse_date(record['dateTime'])
            if start_date <= dt <= end_date:
                filtered.append((dt, float(record['value'])))
        except Exception as e:
            continue
    # Sort by date
    filtered.sort(key=lambda x: x[0])
    dates = [x[0] for x in filtered]
    values = [x[1] for x in filtered]
    return dates, np.array(values)

def extract_all_series(node_data, start_date, end_date, feature_keys):
    """
    For the given node, return a dict {key -> (dates, values)}.
    Missing series → empty list.
    """
    out = {}
    for key in feature_keys:
        raw  = node_data.get(key, [])
        pairs = [
            (parse_date(rec["dateTime"]), float(rec["value"]))
            for rec in raw
            if start_date <= parse_date(rec["dateTime"]) <= end_date
        ]
        pairs.sort(key=lambda x: x[0])
        dates, vals = zip(*pairs) if pairs else ([], [])
        out[key] = (list(dates), np.array(vals, dtype=float))
    return out

def build_node_feature_matrix(G, start_date, end_date, config):
    """
    For each node in graph G, extract its time series values within [start_date, end_date], 
    align them on a common time index (union of all dates), and return:
      - A feature matrix X of shape (num_nodes, T), where T is the number of common time steps.
      - The common sorted time index (list of datetime objects).
      - The list of node identifiers (in the same order as the rows of X).
    Missing values are filled with np.nan.
    """
    nodes = list(G.nodes())
    # Store each node's dates and values, and build the union of all dates.
    node_series = {}
    common_dates = set()
    for node in nodes:
        dates, values = extract_series_from_node(G.nodes[node], start_date, end_date, config)
        node_series[node] = (dates, values)
        common_dates.update(dates)
    
    common_dates = sorted(common_dates)
    
    # Build the feature matrix by aligning each node's series to the common_dates.
    matrix = []
    for node in nodes:
        dates, values = node_series[node]
        date_value_map = {d: v for d, v in zip(dates, values)}
        # if a date is missing for a node, fill with 0
        aligned_values = [date_value_map.get(d, 0) for d in common_dates]
        matrix.append(aligned_values)
    
    X = np.array(matrix)  # shape: (num_nodes, T)
    return X, common_dates, nodes

def build_node_feature_tensor(G, start_date, end_date, config):
    nodes = list(G.nodes())
    keylist = config.feature_keys            # 4 keys
    common_dates = set()

    # first pass – collect each node’s time stamps
    node_series = {}
    for n in nodes:
        node_series[n] = extract_all_series(G.nodes[n], start_date, end_date, keylist)
        # always include discharge dates (mandatory)
        common_dates.update(node_series[n]["discharge_series"][0])

    # convert to sorted list
    common_dates = sorted(common_dates)
    T = len(common_dates)
    num_feats = len(keylist)
    num_nodes = len(nodes)

    tensor = np.zeros((num_nodes, num_feats, T), dtype=np.float32)

    # fill tensor
    for ni, n in enumerate(nodes):
        for fi, key in enumerate(keylist):
            dates, vals = node_series[n][key]
            d2v = {d: v for d, v in zip(dates, vals)}
            tensor[ni, fi, :] = [d2v.get(d, 0.0) for d in common_dates]

    return tensor, common_dates, nodes

def build_dataset(data_matrix, input_len, target_len):
    """
    Given a data matrix of shape (num_nodes, T), create a dataset using a sliding window.
    Each sample:
      - X_sample has shape (num_nodes, input_len)
      - Y_sample has shape (num_nodes, target_len)
    Returns:
      - X_samples: shape (num_samples, num_nodes, input_len)
      - Y_samples: shape (num_samples, num_nodes, target_len)
    Raises a ValueError if there is insufficient data.
    """
    _, T = data_matrix.shape
    num_samples = T - (input_len + target_len) + 1

    if num_samples < 1:
        raise ValueError(
            f"Not enough data points to create sliding windows. "
            f"Time steps (T) = {T} but require at least {input_len + target_len}."
        )
    
    data_tensor = torch.tensor(data_matrix, dtype=torch.float32)

    scaler = NodeStandardScaler().fit(data_tensor)
    scaler.save("node_scaler.pt")

    data_tensor = scaler.transform(data_tensor) 
    
    X_samples = []
    Y_samples = []
    for i in range(num_samples):
        X_sample = torch.tensor(data_tensor[:, i:i + input_len],  dtype=torch.float32)
        Y_sample = torch.tensor(data_tensor[:, i + input_len : i + input_len + target_len],
                         dtype=torch.float32)

        X_samples.append(X_sample)
        Y_samples.append(Y_sample)
    
    X_samples = np.stack(X_samples, axis=0)
    Y_samples = np.stack(Y_samples, axis=0)
    return X_samples, Y_samples

def build_dataset(tensor, input_len, target_len, target_feat_idx=0):
    """
    tensor: (N, F, T)
    Returns
      X  (samples, N, F, input_len)
      Y  (samples, N, target_len)   # predict discharge only (feat 0)
    """
    _, _, T = tensor.shape
    num_samples = T - (input_len + target_len) + 1
    if num_samples < 1:
        raise ValueError(f"Need ≥{input_len+target_len} timesteps, got {T}")

    Xs, Ys = [], []
    for i in range(num_samples):
        Xs.append(tensor[:, :, i:i+input_len])
        Ys.append(tensor[:, target_feat_idx, i+input_len : i+input_len+target_len])
    return np.stack(Xs), np.stack(Ys)

def build_adjacency_matrix(G, nodes_order):
    """
    Build a binary adjacency matrixfor graph G.
    The order of nodes in the matrix is given by nodes_order.
    """
    num_nodes = len(nodes_order)
    A = np.zeros((num_nodes, num_nodes))
    node_to_index = {node: i for i, node in enumerate(nodes_order)}
    for u, v in G.edges():
        i = node_to_index[u]
        j = node_to_index[v]
        A[i, j] = 1
        A[j, i] = 1  # undirected graph
    return A

def build_weighted_adjacency_matrix(G, nodes_order):
    """
    Construct a weighted adjacency matrix where each entry is the normalized distance
    between nodes (if an edge exists) and 0 otherwise.
    
    Args:
        G (networkx.Graph): Graph where each edge has a 'normalized_distance' attribute.
        nodes_order (list): List of node identifiers in the order that will correspond to the rows/columns.
        
    Returns:
        A (np.ndarray): Weighted adjacency matrix of shape (num_nodes, num_nodes)
    """
    num_nodes = len(nodes_order)
    A = np.zeros((num_nodes, num_nodes), dtype=float)
    node_to_index = {node: i for i, node in enumerate(nodes_order)}
    for u, v, data in G.edges(data=True):
        i = node_to_index[u]
        j = node_to_index[v]
        # Use the normalized distance stored as an edge attribute; default to 0.0 if not found.
        normalized_distance = data.get('normalized_distance', 0.0)
        A[i, j] = normalized_distance
        A[j, i] = normalized_distance  # for an undirected graph
    return A

def adj_to_edge_index_and_attr(A):
    """
    Converts an adjacency matrix A (NumPy array) to PyTorch Geometric's edge_index and edge_attr.
    
    Args:
        A (np.ndarray): Adjacency matrix of shape (num_nodes, num_nodes). Can be binary or weighted.
    
    Returns:
        edge_index (torch.LongTensor): Tensor of shape [2, num_edges].
        edge_attr (torch.FloatTensor): Tensor of shape [num_edges] containing edge attributes.
    """
    # Find indices where A is nonzero
    row, col = np.nonzero(A)
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    # Get corresponding edge attributes (if A is binary, these will be ones; if weighted, the actual weights)
    edge_attr = torch.tensor(A[row, col], dtype=torch.float)
    return edge_index, edge_attr

def nash_sutcliffe_efficiency_batch(obs: torch.Tensor,
                              sim: torch.Tensor,
                              eps: float = 1e-10) -> torch.Tensor:
    """
    Compute NSE for every (batch, node) pair.
    obs, sim: shape (B, N, T)   -- T is pred_len
    Returns:  shape (B, N)
    """
    num = torch.sum((sim - obs) ** 2, dim=-1)  # (B, N)
    mean_obs = torch.mean(obs, dim=-1, keepdim=True)                # (B, N, 1)
    denom = torch.sum((obs - mean_obs) ** 2, dim=-1) + eps       # (B, N)

    return 1.0 - num / denom

def nse_nodes_concat_batches(
    obs: torch.Tensor,     # shape (B, N, L)
    sim: torch.Tensor,     # shape (B, N, L)
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Nash-Sutcliffe efficiency for each node after concatenating all batches.

    *   For node j, we stack its B forecast/obs windows of length L end-to-end
        → one long series of length B·L, and compute NSE on that series.
    *   Returns one NSE per node.

    Parameters
    ----------
    obs, sim : torch.Tensor
        Shape (B, N, L)  –  observation and simulation / prediction.
    eps : float
        Small constant to avoid division by zero when variance is zero.

    Returns
    -------
    torch.Tensor
        Shape (N,) – NSE for every node.
    """
    if obs.shape != sim.shape:
        raise ValueError("obs and sim must have the same shape (B, N, L).")

    # --- 1.  reshape so that each row = 1 node's concatenated time series ----
    # permute: (B, N, L) -> (N, B, L)  then flatten last two dims -> (N, B·L)
    obs_flat = obs.permute(1, 0, 2).reshape(obs.size(1), -1)
    sim_flat = sim.permute(1, 0, 2).reshape(sim.size(1), -1)

    # --- 2.  NSE per node -----------------------------------------------------
    mean_obs = obs_flat.mean(dim=1, keepdim=True)                # (N, 1)
    num  = torch.sum((sim_flat - obs_flat) ** 2, dim=1)          # (N,)
    denom = torch.sum((obs_flat - mean_obs) ** 2, dim=1) + eps   # (N,)
    nse = 1.0 - num / denom                                      # (N,)

    return nse

def nash_sutcliffe_efficiency(observed, predicted):
    """
    Compute the Nash-Sutcliffe Efficiency (NSE) metric.
    
    Parameters:
        observed: np.array or torch.Tensor
            The observed (ground truth) data.
        predicted: np.array or torch.Tensor
            The predicted data from the model.
    
    Returns:
        nse (float): Nash-Sutcliffe Efficiency metric.
    """
    if isinstance(observed, torch.Tensor):
        observed = observed.detach().cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
        
    obs_flat = observed.flatten()
    pred_flat = predicted.flatten()
    
    numerator = np.sum((obs_flat - pred_flat) ** 2)
    denominator = np.sum((obs_flat - np.mean(obs_flat)) ** 2)
    
    # Avoid division by zero: if denominator is 0, return NSE of 1.0 if numerator is also 0
    if denominator == 0:
        return 1.0 if numerator == 0 else -np.inf
    
    nse = 1 - numerator / denominator
    return nse

def nse_nodewise(y_hat, y, eps=1e-6):
    ss_res = ((y - y_hat) ** 2).sum(-1)                            # (B, N)
    var    = ((y - y.mean(-1, keepdim=True)) ** 2).sum(-1)         # (B, N)

    # Mean across batch for each node
    ss_res_node = ss_res.mean(0)                                   # (N,)
    var_node    = var.mean(0)                                      # (N,)

    # Compute NSE, guard against tiny denominator
    nse_node = 1.0 - ss_res_node / (var_node + eps)                # (N,)

    # Pick only nodes whose variance is not (almost) zero
    valid = var_node > eps
    idxs  = torch.arange(y.size(1))[valid]

    # Convert to Python list of (node_idx, nse)
    return [(int(i), float(nse_node[i])) for i in idxs]   

def plot_random_result(config, model, val_loader, A = None, show = False, n = 10, ylim = (0, 1)):
    scaler = NodeStandardScaler.load("node_scaler.pt")
    for i in range(n):
        for batch_X, batch_Y in val_loader:
            batch_X = batch_X.to(config.device)
            batch_Y = batch_Y.to(config.device)
            
            pred_norm = model(batch_X, A) if A is not None else model(batch_X)
            pred_raw = inverse_one_feature(pred_norm, scaler, feat_idx=0)
            obs_raw = inverse_one_feature(batch_Y, scaler, feat_idx=0)
            batch_X = scaler.inverse_transform(batch_X)

        sample_idx = random.randint(0,batch_X.shape[0]-1)
        node_idx = random.randint(0,config.num_nodes-1)    


        predicted_future = pred_raw[sample_idx, node_idx, :].detach().cpu().numpy()

        print("predicted_future: ", len(predicted_future))
        ground_truth_future = batch_Y[sample_idx, node_idx, :].detach().cpu().numpy()
        print("ground_truth_future: ", len(ground_truth_future))
        input = batch_X.squeeze(1)[sample_idx, node_idx, 0, :].detach().cpu().numpy()
        # Create time steps for the future predictions.
        time_steps = range(0, len(predicted_future) + len(input))
        print("time_steps: ", len(time_steps))

        plt.figure(figsize=(10, 5))
        plt.plot(time_steps[-len(ground_truth_future):], ground_truth_future, label="Ground Truth")
        plt.plot(time_steps[:len(input)], input, label="Input")
        plt.plot(time_steps[-len(predicted_future):], predicted_future, label="Predicted", linestyle="--")
        plt.xlabel("Future Time Step")
        plt.ylabel(f"{config.series_type}")
        plt.ylim(ylim)
        plt.title(f"Future Predictions for Sample {sample_idx}, Node {node_idx}")
        plt.legend()
        plt.grid(True)
        # Save figures
        if not show:
            plt.tight_layout()
            plt.savefig(f"figures/{config.model_name}/sample_{sample_idx}_node_{node_idx}.png", ) 
        else:
            plt.show()


def plot_random_result_models(config, model_1, model_2, val_loader, A = None, show = False, n = 10, ylim = (0, 1)):
    scaler = NodeStandardScaler.load("node_scaler.pt")
    for i in range(n):
        for batch_X, batch_Y in val_loader:
            batch_X = batch_X.to(config.device)
            batch_Y = batch_Y.to(config.device)
            
            y_pred_1 = model_1(batch_X, A) if A is not None else model_1(batch_X)
            y_pred_2 = model_2(batch_X) if A is not None else model_2(batch_X)

            y_pred_1 = scaler.inverse_transform(y_pred_1)
            y_pred_2 = scaler.inverse_transform(y_pred_2)
            batch_Y = scaler.inverse_transform(batch_Y)
            batch_X = scaler.inverse_transform(batch_X)
            
        sample_idx = random.randint(0,batch_X.shape[0]-1)
        node_idx = random.randint(0,config.num_nodes-1)    


        predicted_future_1 = y_pred_1[sample_idx, node_idx, :].detach().cpu().numpy()
        predicted_future_2 = y_pred_2[sample_idx, node_idx, :].detach().cpu().numpy()

        print("predicted_future_1: ", len(predicted_future_1))
        print("predicted_future_2: ", len(predicted_future_2))
        ground_truth_future = batch_Y[sample_idx, node_idx, :].detach().cpu().numpy()
        print("ground_truth_future: ", len(ground_truth_future))
        input = batch_X.squeeze(1)[sample_idx, node_idx, :].detach().cpu().numpy()
        # Create time steps for the future predictions.
        time_steps = range(0, len(predicted_future_1) + len(input))
        print("time_steps: ", len(time_steps))

        plt.figure(figsize=(10, 5))
        plt.plot(time_steps[-len(ground_truth_future):], ground_truth_future, label="Ground Truth")
        plt.plot(time_steps[:len(input)], input, label="Input")
        plt.plot(time_steps[-len(predicted_future_1):], predicted_future_1, label="Ours", linestyle="--")
        plt.plot(time_steps[-len(predicted_future_2):], predicted_future_2, label="GraphWaveNet", linestyle="--")
        plt.xlabel("Future Time Step")
        plt.ylabel(f"{config.series_type}")
        plt.ylim(ylim)
        plt.title(f"Future Predictions for Sample {sample_idx}, Node {node_idx}")
        plt.legend()
        plt.grid(True)
        # Save figures
        if not show:
            plt.tight_layout()
            plt.savefig(f"figures/both/sample_{sample_idx}_node_{node_idx}.png", ) 
        else:
            plt.show()