import torch

class Config:
    def __init__(self):
        self.seq_len = 30
        self.pred_len = 10
        self.batch_size = 32
        self.num_nodes = 118
        self.num_epochs = 100
        self.warm_up = 3
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.series_type = "discharge_series"
        self.model_name = "STGNN"  
        self.feature_keys = [
            "discharge_series",      # required
            # "precipitation",
            # "total_precipitation",
            "water_temp_celsius",
            # "water_temp_fahrenheit",
            # "evaporation_temp_48",
            # "evaporation_temp_24",
            "air_temp_celsius",
            # "air_temp_fahrenheit",
            # "solar_radiation",
            # "cloud_cover",
            # "wind_speed",
            # "evaporation_rate",
            # "relative_humidity",
        ]
        self.num_feats = len(self.feature_keys)
