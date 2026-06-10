import os
import hydra
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader

from data.transform_data import DataBuilder

from common.utils import split_train_valid_test_on_id_clients, balance_data
from common.datasets import TSDataset
from common.metrics import ImbalancedClassificationMetrics
from common.classifier_trainer import BaseClassifierTrainer

from model.framework.adf import ADF
from model.backbone.TransApp import TransAppClassif, TransAppConfig
from model.backbone.TransAppV2 import TransAppV2Classif, TransAppV2Config

from dataclasses import asdict
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_appliance_label(cfg, data):
    """
    Load appliance label 
    """
    logging.info(f"Load labels for appliance: {cfg.finetuning.appliance}")
    if cfg.finetuning.appliance in ['ac', 'heater', 'heater_type', 'pac', 'pool']:
        if cfg.finetuning.appliance == 'ac' or cfg.finetuning.appliance == 'pool':  # Keep Only Subsequences Between June and September (Summer)
            data = data.loc[data['start_date'].dt.month >= 6]
            data = data.loc[data['start_date'].dt.month < 9]
        elif cfg.finetuning.appliance == 'heater_type' or cfg.finetuning.appliance == 'pac': # Keep Only Subsequences Between November and March (Winter)
            boolean_mask = (data['start_date'].dt.month > 3) & (data['start_date'].dt.month < 11)
            data = data.loc[~boolean_mask]
        elif cfg.finetuning.appliance == 'heater': # Keep Only Subsequences Between December and February
            boolean_mask = (data['start_date'].dt.month > 2) & (data['start_date'].dt.month < 12)
            data = data.loc[~boolean_mask]

    labels = pd.read_parquet(f'{cfg.data.label_path}/label_{cfg.finetuning.appliance}.parquet')
    data = pd.merge(data, labels, on=cfg.data.id_name)

    return data


def from_pretrained(path_pretrain_core: str):

    logging.info(f"Loading state dict from {path_pretrain_core}")
    log = torch.load(path_pretrain_core, map_location='cpu', weights_only=False)

    if 'TransAppVersion' in log:
        if log['TransAppVersion']=='1':
            cfg = TransAppConfig(**log['config'])
            model = TransAppClassif(cfg)
            model.load_state_dict(log['model_state_dict'], strict=True)

        elif log['TransAppVersion']=='2':
            cfg = TransAppV2Config(**log['config'])
            model = TransAppV2Classif(cfg)
            model.load_state_dict(log['model_state_dict'], strict=True)

        else:
            raise ValueError(f"TransApp version {log['version']} unknwon.")

    else:
        cfg = TransAppV2Config(**log['config'])
        model = TransAppV2Classif(cfg)
        model.load_state_dict(log['model_state_dict'], strict=True)

    return model, cfg
 

def train_cross_validation(cfg, data, seed):
    
    df_train, df_valid, df_test = split_train_valid_test_on_id_clients(data, 
                                                                       valid_size=0.2, test_size=0.2, 
                                                                       id_clients=cfg.data.id_name, 
                                                                       seed=seed)
    df_train = balance_data(df_train)

    train_dataset = TSDataset(df_train, 
                              exogene_var=cfg.exogene_variable,
                              id_clients=cfg.data.id_name,
                              freq=cfg.data.sampling_rate,
                              id_label=cfg.data.col_label_name,
                              scaling_method=cfg.scaling.scaler_type,
                              scale_param1=cfg.scaling.scale_param1,
                              scale_param2=cfg.scaling.scale_param2
                            )
    
    valid_dataset = TSDataset(df_valid, 
                              exogene_var=cfg.exogene_variable,
                              id_clients=cfg.data.id_name,
                              freq=cfg.data.sampling_rate,
                              id_label=cfg.data.col_label_name,
                              scaling_method=cfg.scaling.scaler_type,
                              scale_param1=cfg.scaling.scale_param1,
                              scale_param2=cfg.scaling.scale_param2
                            )
    
    test_dataset = TSDataset(df_test, 
                              exogene_var=cfg.exogene_variable,
                              id_clients=cfg.data.id_name,
                              freq=cfg.data.sampling_rate,
                              id_label=cfg.data.col_label_name,
                              scaling_method=cfg.scaling.scaler_type,
                              scale_param1=cfg.scaling.scale_param1,
                              scale_param2=cfg.scaling.scale_param2
                            )

    train_loader = DataLoader(train_dataset, batch_size=cfg.finetuning.batch_size, shuffle=True, num_workers=cfg.finetuning.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.finetuning.batch_size, shuffle=False, num_workers=cfg.finetuning.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.finetuning.batch_size, shuffle=False, num_workers=cfg.finetuning.num_workers)

    model, model_cfg = from_pretrained(cfg.finetuning.path_pretrained_weights)

    trainer = BaseClassifierTrainer(
        model,
        train_loader,
        valid_loader,
        optimizer_kwargs = {"lr": cfg.finetuning.lr, 
                            "weight_decay": cfg.finetuning.wd},
        lr_scheduler_kwargs = {"mode": "min", "patience": 5, "eps": 1e-7},
        criterion = nn.CrossEntropyLoss(),
        patience_es = cfg.finetuning.patience_es,
        device = cfg.finetuning.device,
        use_data_parallel = cfg.finetuning.use_data_parallel,
        n_warmup_epochs = cfg.finetuning.n_warmup_epochs,
        metrics = ImbalancedClassificationMetrics(),
        save_checkpoint = True,
        checkpoint_path = f"models/{cfg.finetuning.appliance}/{cfg.finetuning.modelname}_{cfg.finetuning.seed}",
        verbose = True,
    )

    trainer.train(cfg.finetuning.epochs)
    trainer.restore_best_weights(tag="best")

    # trainer.restore_best_weights()
    trainer.evaluate(test_loader)

    voter = ADF(trainer.model, 
                average_mode='quantile',
                classif_metrics=ImbalancedClassificationMetrics(),
                dataset_kwargs={"exogene_var": cfg.exogene_variable,
                                "id_clients": cfg.data.id_name,
                                "freq": cfg.data.sampling_rate,
                                "id_label": "label"},
                device=cfg.finetuning.device,
                batch_size_voter=cfg.finetuning.batch_size,
    )

    voter.train(df_valid, monitoring_metric="F1_MACRO")
    metrics = voter.test(df_test)
    logging.info(metrics)

    metrics = {'appliance': cfg.finetuning.appliance,
               'quantile': voter.quantile,
               'seed': cfg.finetuning.seed,
               'accuracy': metrics['ACCURACY'], 
               'balanced_accuracy': metrics['BALANCED_ACCURACY'], 
               'precision': metrics['PRECISION'], 
               'recall': metrics['RECALL'], 
               'f1': metrics['F1'], 
               'f1macro': metrics['F1_MACRO']
               }
    
    for tag in ["best", "final"]:
        os.remove(f"models/{cfg.finetuning.appliance}/{cfg.finetuning.modelname}_{cfg.finetuning.seed}_{tag}")
        logging.info(f"Delete 'models/{cfg.finetuning.appliance}/{cfg.finetuning.modelname}_{cfg.finetuning.seed}_{tag}'")

    return metrics


def train_final(cfg, data):
    
    df_train, df_valid = split_train_valid_test_on_id_clients(data, 
                                                              test_size=0.1, # for monitoring final training
                                                              id_clients=cfg.data.id_name, 
                                                              seed=42)
    df_train = balance_data(df_train)

    train_dataset = TSDataset(df_train, 
                              exogene_var=cfg.exogene_variable,
                              id_clients=cfg.data.id_name,
                              freq=cfg.data.sampling_rate,
                              id_label=cfg.data.col_label_name,
                              scaling_method=cfg.scaling.scaler_type,
                              scale_param1=cfg.scaling.scale_param1,
                              scale_param2=cfg.scaling.scale_param2
                            )
    
    valid_dataset = TSDataset(df_valid, 
                              exogene_var=cfg.exogene_variable,
                              id_clients=cfg.data.id_name,
                              freq=cfg.data.sampling_rate,
                              id_label=cfg.data.col_label_name,
                              scaling_method=cfg.scaling.scaler_type,
                              scale_param1=cfg.scaling.scale_param1,
                              scale_param2=cfg.scaling.scale_param2
                            )
    

    train_loader = DataLoader(train_dataset, batch_size=cfg.finetuning.batch_size, shuffle=True, num_workers=cfg.finetuning.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.finetuning.batch_size, shuffle=False, num_workers=cfg.finetuning.num_workers)

    model, model_cfg = from_pretrained(cfg.finetuning.path_pretrained_weights)

    trainer = BaseClassifierTrainer(
        model,
        train_loader,
        valid_loader,
        optimizer_kwargs = {"lr": cfg.finetuning.lr, 
                            "weight_decay": cfg.finetuning.wd},
        lr_scheduler_kwargs = {"mode": "min", "patience": 5, "eps": 1e-7},
        criterion = nn.CrossEntropyLoss(),
        patience_es = cfg.finetuning.patience_es,
        device = cfg.finetuning.device,
        use_data_parallel = cfg.finetuning.use_data_parallel,
        n_warmup_epochs = cfg.finetuning.n_warmup_epochs,
        metrics = ImbalancedClassificationMetrics(),
        save_checkpoint = True,
        checkpoint_path = f"models/{cfg.finetuning.appliance}/{cfg.finetuning.modelname}",
        verbose = True
    )

    trainer.train(cfg.finetuning.epochs)
    trainer.restore_best_weights(tag="best")

    voter = ADF(trainer.model, 
                average_mode='quantile',
                classif_metrics=ImbalancedClassificationMetrics(),
                dataset_kwargs={"exogene_var": cfg.exogene_variable,
                                "id_clients": cfg.data.id_name,
                                "freq": cfg.data.sampling_rate,
                                "id_label": "label"},
                device=cfg.finetuning.device,
                batch_size_voter=cfg.finetuning.batch_size,
    )

    cv_res = pd.read_csv(os.path.join(cfg.finetuning.path_results, f"{cfg.finetuning.modelname}.csv"))
    quantile = cv_res.groupby('appliance').mean().loc[cfg.finetuning.appliance]['quantile']
    voter.quantile = quantile
    voter.is_fitted = True

    metrics = voter.test(df_valid)
    logging.info('Metrics on monitoring valid dataset')
    logging.info(metrics)
    
    os.remove(f"models/{cfg.finetuning.appliance}/{cfg.finetuning.modelname}_final")

    log = {}
    log["appliance"] = cfg.finetuning.appliance
    log["subsequence_length"] = cfg.data.win
    log["sampling_rate"] = "30min"
    log["adf_quantile"] = float(round(quantile, 2))
    log["transapp_version"] = cfg.finetuning.modelname[-1:]
    log["config"] = asdict(model.config) if hasattr(model, "config") else model_cfg
    log["exogene_variable"] = cfg.exogene_variable
    log["model_state_dict"] = trainer.log["model_state_dict"]

    torch.save(log,  f"models/{cfg.finetuning.appliance}/{cfg.finetuning.modelname}_base_{cfg.finetuning.appliance}.ckpt")

    return


def _infer_appliance_columns(cfg: DictConfig, data: pd.DataFrame) -> list[str]:
    """
    Try to recover the appliance label columns from the config first.
    If none are found, fall back to inferring binary columns from the dataframe.
    """
    config_paths = [
        "data.appliances",
        "appliances",
        "labels.appliances",
        "finetuning.appliances",
    ]

    for path in config_paths:
        appliances = OmegaConf.select(cfg, path)
        if appliances is not None:
            return [col for col in appliances if col in data.columns]

    protected_cols = {
        cfg.data.id_name,
        cfg.data.timestamp_name,
        cfg.data.power_name,
    }

    inferred = []
    for col in data.columns:
        if col in protected_cols:
            continue

        non_null = data[col].dropna()
        if non_null.empty:
            continue

        unique_values = set(pd.unique(non_null))
        if unique_values.issubset({0, 1, False, True}):
            inferred.append(col)

    return inferred


def _collect_dataset_stats(cfg: DictConfig, data: pd.DataFrame) -> dict:
    """
    Collect dataset-level statistics to append to the CV results CSV:
    - number of unique IDs
    - number of 0/1 for each appliance label column
    """
    stats = {
        "data_n_unique_ids": int(data[cfg.data.id_name].nunique())
    }

    appliance_cols = _infer_appliance_columns(cfg, data)
    for appliance in appliance_cols:
        series = data[appliance].dropna()
        stats[f"{appliance}_n_0"] = int((series == 0).sum())
        stats[f"{appliance}_n_1"] = int((series == 1).sum())

    return stats


@hydra.main(version_base=None, config_path="config", config_name="TransAppV2")
def main(cfg: DictConfig) -> None:
    logging.info(cfg)
    logging.info("Reading parquet file...")

    processed_dir = os.path.join(cfg.data.processed_data_path, f"data_{cfg.data.win}")
    candidate = os.path.join(processed_dir, "data.parquet")

    if os.path.exists(candidate):
        load_curves = pd.read_parquet(candidate)
    else:
        load_curves = pd.read_parquet(cfg.data.raw_data_path)

        data_builder: DataBuilder = DataBuilder(
            window_size=cfg.data.win,
            window_stride=cfg.data.win,
            sampling_rate="30min",
            limit_ffill=cfg.data.limit_ffill,
            id_name=cfg.data.id_name,
            timestamp_name=cfg.data.timestamp_name,
            power_name=cfg.data.power_name,
        )

        data_builder.check_input(load_curves)
        load_curves = data_builder.missing_data(load_curves.copy())
        load_curves = data_builder.transform_data(load_curves)
        load_curves = load_curves.reset_index()

        os.makedirs(processed_dir, exist_ok=True)
        load_curves.to_parquet(candidate, index=False)
        logging.info("Processed parquet saved to %s", candidate)

    logging.info(load_curves)

    n_unique_ids = load_curves[cfg.data.id_name].nunique()
    print("Number of unique IDs:", n_unique_ids)

    data = get_appliance_label(cfg, load_curves)

    print(data.head())
    print(data.shape)

    n_unique_ids = data[cfg.data.id_name].nunique()
    print("Number of unique IDs:", n_unique_ids)

    dataset_stats = _collect_dataset_stats(cfg, data)

    if cfg.finetuning.cross_validation:
        logging.info("Perform cross-validation training...")
        metrics = train_cross_validation(cfg, data, cfg.finetuning.seed)

        logging.info("Save results for seed %s...", cfg.finetuning.seed)

        results_dir = os.path.join(cfg.finetuning.path_results, str(cfg.data.win))
        os.makedirs(results_dir, exist_ok=True)

        csv_path = os.path.join(
            results_dir,
            f"{cfg.finetuning.modelname}.csv",
        )

        row_to_save = {**metrics, **dataset_stats}
        write_header = not os.path.exists(csv_path)
        pd.DataFrame([row_to_save]).to_csv(
            csv_path,
            mode="a",
            header=write_header,
            index=False,
        )
    else:
        logging.info("Train final model...")
        train_final(cfg, data)

if __name__ == "__main__":
    main()
