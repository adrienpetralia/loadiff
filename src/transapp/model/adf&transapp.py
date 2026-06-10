import logging
import warnings
import tqdm
from typing import Optional, Union, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data.transform_data import DataBuilder

from common.datasets import TSDataset

from model.backbone.TransApp import TransAppClassif, TransAppConfig
from model.backbone.TransAppV2 import TransAppV2Classif, TransAppV2Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ADFTransAppInference:
    def __init__(
        self,
        cfg,
        device: str = "cuda"
    ) -> None:

        self.cfg = cfg

        if ("cuda" in device) and (not torch.cuda.is_available()):
            device = torch.device("cpu")
            warnings.warn(f"Device {device} selected but cuda not detected not available, device set to 'cpu'.")
        
        else:
            device = torch.device(device)
            logging.info(f"{device} selected.")

        self.device = device

        super().__init__()

    def from_pretrained(self, path_pretrain_model: str):

        self.log = torch.load(path_pretrain_model, map_location='cpu', weights_only=False)

        if self.log['transapp_version']=='1':
            transapp_cfg = TransAppConfig(**self.self.log['config'])
            model = TransAppClassif(transapp_cfg)
            model.load_state_dict(self.log['model_state_dict'], strict=True)

        elif self.log['transapp_version']=='2':
            transapp_cfg = TransAppV2Config(**self.log['config'])
            model = TransAppV2Classif(transapp_cfg)
            model.load_state_dict(self.log['model_state_dict'], strict=True)

        else:
            raise ValueError(f"TransApp version '{self.log['transapp_version']}' doesn't match a known version '1,2'.")

        return model

    def _check_input(self, data):

        if (len(data.shape)!=2) or (data.shape[1]!=3):
           raise ValueError("Input data shape doesn't match required long time series data format.")
        
        return

    def transform_long_df_to_subsequences(self, long_df):
        
        data_builder: DataBuilder = DataBuilder(window_size=self.log['subsequence_length'],
                                                window_stride=self.log['subsequence_length'],
                                                sampling_rate=self.log['sampling_rate'],
                                                id_name=self.cfg.data.id_name,
                                                timestamp_name=self.cfg.data.timestamp_name,
                                                power_name=self.cfg.data.power_name)

        logger.info("checking input...")
        data_builder.check_input(long_df)
        long_df = data_builder.missing_data(long_df.copy())
        logger.info("Processing data...")
        long_df = data_builder.transform_data(long_df)
        logger.debug(f'data processed :\n{long_df}')
        load_curves = load_curves.reset_index()
        logging.info(load_curves)

        return load_curves
    
    def get_appliance_period(self, 
                            data: pd.Dataframe, 
                            appliance: str):
        """
        Load appliance label 
        """
        if appliance in ['ac', 'heater', 'heater_type']:
            if appliance == 'ac':  # Keep Only Subsequences Between June and September (Summer)
                data = data.loc[data['start_date'].dt.month >= 6]
                data = data.loc[data['start_date'].dt.month < 9]
            elif appliance == 'heater_type': # Keep Only Subsequences Between November and March (Winter)
                boolean_mask = (data['start_date'].dt.month > 3) & (data['start_date'].dt.month < 11)
                data = data.loc[~boolean_mask]
            elif appliance == 'heater': # Keep Only Subsequences Between December and February
                boolean_mask = (data['start_date'].dt.month > 2) & (data['start_date'].dt.month < 12)
                data = data.loc[~boolean_mask]

        return data

    def __call__(self, 
                 data: pd.DataFrame,
                 appliances: list[str] = ["ev"],
                 batch_size: int = 1) -> pd.DataFrame:
        

        data = self.transform_long_df_to_subsequences(data)

        logger.info("Starting predictions...")
        prediction = data[[self.cfg.data.id_name]]

        for appliance in appliances:
            model = self.from_pretrained("path_to_pretrained_model")
            model.to(self.device)
            self.model.eval()

            # TODO: get_appliance_label: obligatoire ou que chuaffage dan tous les cas ? 
            data_app = self.get_appliance_period(data, appliance) # TODO: que doit on retourer pour les personnes exclus (chauffage, conv/pac) ?

            with torch.inference_mode():
                dl = torch.utils.data.DataLoader(
                    TSDataset(data_app, 
                              exogene_var=self.log["exogene_variable"],
                              id_clients=self.cfg.data.id_name,
                              freq=self.cfg.data.sampling_rate
                            ),
                    batch_size=batch_size,
                )

                pred_one_app = []
                for ts, exogene in dl:
                    logits = self.model(ts.float().to(self.device), exogene.float().to(self.device))
                    pred_one_app.append(nn.Softmax(dim=1)(logits)[:, 1].cpu().numpy().ravel())

            prediction[appliance] = pred_one_app

        prediction = prediction.groupby(self.cfg.data.id_name).mean() # TODO: facon plus intelligente qui prend en compte le quantile tuné pour chaque appareil ?

        return prediction