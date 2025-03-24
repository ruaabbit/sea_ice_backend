import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import lightning as L


from .config import configs
from .dataset import SIC_dataset
from .utils.model_factory import Network
from .utils.metrics import *

# Datasets and Dataloaders
dataset_train = SIC_dataset(
    configs.full_data_path,
    configs.train_period[0],
    configs.train_period[1],
    configs.input_gap,
    configs.input_length,
    configs.pred_shift,
    configs.pred_gap,
    configs.pred_length,
    samples_gap=1,
)

dataset_vali = SIC_dataset(
    configs.full_data_path,
    configs.eval_period[0],
    configs.eval_period[1],
    configs.input_gap,
    configs.input_length,
    configs.pred_shift,
    configs.pred_gap,
    configs.pred_length,
    samples_gap=1,
)

dataloader_train = DataLoader(
    dataset_train,
    batch_size=configs.batch_size,
    shuffle=True,
    num_workers=configs.num_workers,
)

dataloader_vali = DataLoader(
    dataset_vali,
    batch_size=configs.batch_size_vali,
    shuffle=False,
    num_workers=configs.num_workers,
)


# Define PyTorch Lightning Module
class MyLightningModule(L.LightningModule):
    def __init__(self):
        super(MyLightningModule, self).__init__()
        self.network = Network()
        self.mask = torch.from_numpy(np.load("seaice/cross_modality/data/arctic_mask.npy"))
        self.save_hyperparameters(configs.__dict__)

    def forward(self, inputs, inputs_mark, targets_mark):
        return self.network(inputs, inputs_mark, targets_mark)

    def _calculate_metrics(self, pred, targets, dataset, prefix=""):
        mask = self.mask.to(self.device).float()

        # 计算 loss 时不进行反归一化
        loss = loss_func(pred[:, :, 0, :, :], targets[:, :, 0, :, :], mask)
        # 计算其他指标时进行反归一化
        pred_sic = dataset._denormalize(pred[:, :, 0, :, :], 0)
        targets_sic = dataset._denormalize(targets[:, :, 0, :, :], 0)
        # SIC大于等于0.15的才算
        pred_sic = pred_sic * (pred_sic >= 0.15)
        targets_sic = targets_sic * (targets_sic >= 0.15)

        metrics = {
            f"{prefix}loss": loss,
            f"{prefix}mae_sic": mae_func(pred_sic, targets_sic, mask),
            f"{prefix}rmse_sic": rmse_func(pred_sic, targets_sic, mask),
            f"{prefix}R^2_sic": r2_func(pred_sic, targets_sic, mask),
        }
        return metrics

    def training_step(self, batch):
        inputs, targets, inputs_mark, targets_mark = batch
        pred = self.network(inputs, inputs_mark, targets_mark)
        metrics = self._calculate_metrics(pred, targets, dataset_train)
        self.log_dict(metrics, prog_bar=True, logger=False, sync_dist=True)
        return metrics["loss"]

    def validation_step(self, batch):
        inputs, targets, inputs_mark, targets_mark = batch
        pred = self.network(inputs, inputs_mark, targets_mark)
        metrics = self._calculate_metrics(pred, targets, dataset_vali, prefix="val_")
        self.log_dict(metrics, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(), lr=configs.lr)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=configs.lr,
            epochs=configs.num_epochs,
            steps_per_epoch=len(dataloader_train),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update the learning rate after every optimizer step
            },
        }
