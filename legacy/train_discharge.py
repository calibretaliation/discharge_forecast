
from dataset import prepare_data
from config import Config
from train import *
from utils import *
import argparse

argsparser = argparse.ArgumentParser(description="Train a model for time series forecasting.")
argsparser.add_argument("--model_name", type=str, default="STGAT", help="Name of the model to train.")
argsparser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train the model.")
argsparser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
argsparser.add_argument("--seq_len", type=int, default=30, help="Length of the input sequence.")
argsparser.add_argument("--pred_len", type=int, default=10, help="Length of the prediction sequence.")
argsparser.add_argument("--device", type=str, default="cuda:1", help="Device to use for training (e.g., 'cuda:0' or 'cpu').")
argsparser.add_argument("--graph_filename", type=str, default="data/usgs_graph_filtered_multivar.gpickle", help="Path to the graph file.")

if __name__ == "__main__":
    args = argsparser.parse_args()
    print("Arguments:", args)
    config = Config()
    config.series_type = "discharge_series"
    config.model_name = args.model_name
    config.num_epochs = args.num_epochs
    config.warm_up = 3
    config.batch_size = args.batch_size
    config.seq_len = args.seq_len
    config.pred_len = args.pred_len
    config.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, supports, edge_index, A = prepare_data(config, args.graph_filename)
    if config.model_name == "graphwavenet":
        A = None
        model, criterion, optimizer = instantiate_model(config, supports)
    else:
        model, criterion, optimizer = instantiate_model(config, edge_index)
    nse_values = train_model(config, model, optimizer, criterion, train_loader, val_loader, config.num_epochs, edge_index)
    np.save(f"nse_values_{args.model_name}.npy", nse_values)
    plot_random_result(config, model, val_loader, A, show = False, n = 10, ylim = None)
    graph_filename = "data/usgs_graph_filtered_discharge.gpickle"
    
