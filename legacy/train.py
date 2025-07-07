import random
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import prepare_data
from graph_wavenet import gwnet
from model import *
from utils import *

def instantiate_model(config, supports):
    # Load the model
    if config.model_name == "graphwavenet":
        model = gwnet(
        device=config.device,
        num_nodes=config.num_nodes,
        dropout=0.3,
        supports=supports,
        gcn_bool=True,
        addaptadj=True,
        aptinit=None,
        in_dim=config.num_feats,
        out_dim=config.pred_len,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=8,
        layers=2
    ).to(config.device)
    else:
        model = NhatModel(num_nodes=config.num_nodes, 
                          in_feat=config.num_feats, 
                          hid_feat=64, 
                          num_blocks=4, 
                          T = config.seq_len, 
                          T_hat=config.pred_len).to(config.device)
    criterion = nn.MSELoss()
    # criterion = NashSutcliffeLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    return model, criterion, optimizer

class NashSutcliffeLoss(nn.Module):
    """
    Differentiable −NSE loss.
    Expects tensors of shape (B, N, L) or (B, L) etc.
    """
    def __init__(self, eps: float = 1e-6, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        y_hat, y : identical shape, any leading batch dims.
        """
        # dimensions to reduce over (time axis = last dim)
        time_dim = -1

        # numerator = SSR = Σ (y - ŷ)²
        ss_res = torch.sum((y - y_hat) ** 2, dim=time_dim)

        # denominator = SST = Σ (y - ȳ)²   (ȳ over time dim)
        y_mean = torch.mean(y, dim=time_dim, keepdim=True)
        ss_tot = torch.sum((y - y_mean) ** 2, dim=time_dim)

        nse = 1.0 - ss_res / (ss_tot + self.eps)   # avoid divide-by-0
        loss = -nse                                # maximise NSE → minimise −NSE

        if self.reduction == "mean":
            return loss.mean()     # average over batch & nodes
        elif self.reduction == "sum":
            return loss.sum()
        else:                       # "none"
            return loss

    # with tqdm(range(config.num_epochs), colour= "red", position = 0, leave = True) as epochs:
    #     for epoch in epochs:
def train_epoch(config, model, optimizer, criterion, train_loader, scaler, A = None):
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    with tqdm(train_loader, colour = "green", position = 1, leave = False) as tepoch:
        model.train()
        total_train_loss = 0.0
        for batch_X, batch_Y in tepoch:
            batch_X = batch_X.to(config.device)  # shape: [batch_size, num_nodes, num_feats, input_seq_len]
            batch_Y = batch_Y.to(config.device)  # shape: [batch_size, num_nodes, num_feats, target_seq_len]
            
            optimizer.zero_grad()

            y_pred = model(batch_X, A) if A is not None else model(batch_X)

            loss = criterion(y_pred, batch_Y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            total_train_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        return avg_train_loss
def validate_epoch(model, val_loader, scaler, criterion,
                   config, A=None, eps=1e-6):
    """
    Returns
    -------
    avg_val_loss : float
    nse_mean     : float                      # mean over valid nodes
    nse_values   : list[float] length=N       # per-node NSE (nan = constant node)
    """
    model.eval()
    total_val_loss = 0.0

    ss_res_sum, ss_tot_sum = None, None   # will be tensors (N,)
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X = batch_X.to(config.device)
            batch_Y = batch_Y.to(config.device)                  # (B,N,T)

            # ── forward ─────────────────────────────────────────────
            pred_norm = model(batch_X, A) if A is not None else model(batch_X)
            pred_raw = inverse_one_feature(pred_norm, scaler, feat_idx=0)
            obs_raw = inverse_one_feature(batch_Y, scaler, feat_idx=0)

            # loss in raw space
            total_val_loss += criterion(pred_norm, batch_Y).item()

            # ── accumulate SSR & SST per node ───────────────────────
            if ss_res_sum is None:                 # allocate once
                N = obs_raw.size(1)
                ss_res_sum = torch.zeros(N, device=config.device)
                ss_tot_sum = torch.zeros(N, device=config.device)

            ss_res_sum += ((obs_raw - pred_raw) ** 2).sum(dim=(0, 2))

            obs_mean = obs_raw.mean(dim=2, keepdim=True)
            ss_tot_sum += ((obs_raw - obs_mean) ** 2).sum(dim=(0, 2))

    # ── final NSE per node ──────────────────────────────────────────
    valid_mask = ss_tot_sum > eps
    nse_nodes = torch.full_like(ss_tot_sum, float("nan"))
    nse_nodes[valid_mask] = 1.0 - ss_res_sum[valid_mask] / (ss_tot_sum[valid_mask] + eps)

    nse_mean = torch.nanmean(nse_nodes).item()           # average over valid nodes
    nse_values = nse_nodes.cpu().tolist()                # Python list, len = N
    # print(nse_values)
    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss, nse_mean, nse_values

def train_model(config, model, optimizer, criterion, train_loader, val_loader, num_epochs, A = None):
    best_val_nse = -1e8
    best_model = None
    scaler = NodeStandardScaler3D.load("node_scaler.pt")
    with tqdm(range(config.num_epochs), colour= "red", position = 0, leave = True) as epochs:
        for epoch in epochs:        
            A = torch.Tensor(A).to(config.device) if A is not None else None
            avg_train_loss = train_epoch(config, model, optimizer, criterion, train_loader, scaler, A)
            avg_val_loss, nse, nse_values = validate_epoch(model, val_loader, scaler, criterion,
                   config, A, eps=1e-3)
            # Save the best model based on validation loss
            if nse > best_val_nse:
                best_val_nse = nse
                best_model = model.state_dict()
                # if epoch > config.warm_up:
                print("Best model at epoch {}/{}\nVal loss: {}\nNSE: {}".format(epoch, num_epochs, avg_val_loss, nse))
            epochs.set_postfix(train_loss=avg_train_loss, val_loss = avg_val_loss, val_nse = nse)
    # Save the model
    torch.save(best_model, "models/{}_{:.2f}.pth".format(config.model_name, best_val_nse))
    print("Model saved as models/{}_{:.2f}.pth".format(config.model_name, best_val_nse))
    return nse_values

