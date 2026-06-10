import hydra
import torch
import pandas as pd

from data.transform_data import DataBuilder

from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from torch.utils.data import DataLoader

from common.utils import split_train_valid_test_on_id_clients
from common.datasets import TSDataset
from common.self_pretrainer import SelfPretrainer

from model.backbone.TransApp import TransAppPretrain, TransAppConfig
from model.backbone.TransAppV2 import TransAppV2Pretrain, TransAppV2Config

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def launch_pretraining(cfg: DictConfig, data, ckpt_path):

    df_train, df_valid = split_train_valid_test_on_id_clients(data, 
                                                              test_size=0.2,
                                                              id_clients=cfg.data.id_name, 
                                                              seed=42)

    train_dataset = TSDataset(df_train, 
                              exogene_var=cfg.exogene_variable,
                              id_clients=cfg.data.id_name,
                              freq=cfg.data.sampling_rate,
                              scaling_method=cfg.scaling.scaler_type,
                              scale_param1=cfg.scaling.scale_param1,
                              scale_param2=cfg.scaling.scale_param2
                            )
    
    valid_dataset = TSDataset(df_valid, 
                              exogene_var=cfg.exogene_variable,
                              id_clients=cfg.data.id_name,
                              freq=cfg.data.sampling_rate,
                              scaling_method=cfg.scaling.scaler_type,
                              scale_param1=cfg.scaling.scale_param1,
                              scale_param2=cfg.scaling.scale_param2
                            )

    train_loader = DataLoader(train_dataset, batch_size=cfg.pretraining.batch_size, shuffle=True, num_workers=cfg.pretraining.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.pretraining.batch_size, shuffle=True, num_workers=cfg.pretraining.num_workers)

    # TODO: add TransAppV1
    model = TransAppV2Pretrain(TransAppV2Config(n_exogene_var=len(cfg.exogene_variable), n_encoder_layers=3))

    trainer = SelfPretrainer(model,                                     
                            train_loader,
                            valid_loader,
                            loss_in_model=True,
                            optimizer_kwargs = {"lr": cfg.pretraining.lr, 
                                                "weight_decay": cfg.pretraining.wd},
                            scheduler_name = cfg.pretraining.scheduler_name,
                            scheduler_kwargs = {"num_warmup_steps": int(len(train_loader) * cfg.pretraining.n_warmup_epochs)},
                            patience_es = cfg.pretraining.patience_es,
                            n_warmup_epochs = cfg.pretraining.n_warmup_epochs,
                            device = cfg.pretraining.device,
                            use_data_parallel = cfg.pretraining.use_data_parallel,
                            checkpoint_path = ckpt_path,
                        )
    
    trainer.train(cfg.pretraining.epochs)


@hydra.main(version_base=None, config_path="config", config_name="TransAppV2")
def main(cfg: DictConfig) -> None:
    logging.info(cfg)

    logging.info("reading parquet file...")
    load_curves = pd.read_parquet(cfg.data.data_path)
    data_builder: DataBuilder = DataBuilder(window_size=cfg.data.win,
                                            window_stride=cfg.data.win,
                                            sampling_rate=cfg.data.sampling_rate,
                                            limit_ffill=cfg.data.limit_ffill,
                                            id_name=cfg.data.id_name,
                                            timestamp_name=cfg.data.timestamp_name,
                                            power_name=cfg.data.power_name)

    data_builder.check_input(load_curves)
    load_curves = data_builder.missing_data(load_curves.copy())
    load_curves = data_builder.transform_data(load_curves)
    load_curves = load_curves.reset_index()
    logging.info(load_curves)

    ckpt_path = f"models/pretrained_weights/{cfg.pretraining.modelname}"
    logging.info(f"Pretrained model will be save at {ckpt_path}")

    logging.info("Start the training...")
    launch_pretraining(cfg, load_curves, ckpt_path)

if __name__ == "__main__":
    main()
