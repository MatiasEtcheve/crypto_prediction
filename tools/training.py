from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics as metrics
import wandb
from tqdm import tqdm

from . import inspect_code


class SaveOutput:
    """Class to save in the forward hook"""

    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


class ScriptCheckpoint(pl.callbacks.Callback):
    def __init__(self, dirpath):
        super().__init__()
        self.dirpath = Path(dirpath)

    def on_pretrain_routine_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        super().on_pretrain_routine_start(trainer, pl_module)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        with open(Path(self.dirpath) / "model_script.txt", "w") as file:
            model_script = inspect_code.get_class_code(type(pl_module))
            file.write(model_script)
        with open(Path(self.dirpath) / "generator_script.txt", "w") as file:
            generator_script = inspect_code.get_class_code(type(pl_module.generator))
            file.write(generator_script)
        with open(Path(self.dirpath) / "discriminator_script.txt", "w") as file:
            discriminator_script = inspect_code.get_class_code(
                type(pl_module.discriminator)
            )
            file.write(discriminator_script)
        filename_datamodule = Path(self.dirpath) / "datamodule_script.txt"
        with open(filename_datamodule, "w") as file:
            file.write(inspect_code.get_class_code(type(trainer.datamodule)))


def train(model, device, train_loader, optimizer, criterion) -> List:
    """Train the model for 1 epoch

    Args:
        model: model to train
        device: device of the training
        train_loader: training dataloader
        optimizer: optimizer of the training
        criterion: loss of the model

    Returns:
        List: list of training loss
    """
    model.train()
    train_loss = []
    for data, target in tqdm(train_loader, total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    return train_loss


def validate(model, device, val_loader, criterion, scaler=None) -> Tuple:
    """Validate the model on the validation dataloader

    Args:
        model: model to evaluate
        device: device of the evaluation
        val_loader: validation dataloader
        criterion: loss of the model
        scaler: Scaler of the data. Defaults to None.

    Returns:
        Tuple: list of validation loss, example images and error
    """
    model.eval()
    val_loss = []
    example_images = []
    error = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss.append(criterion(output, target).item())
            if scaler is not None:
                output = scaler.inverse_transform(output.cpu().numpy())
                target = scaler.inverse_transform(target.cpu().numpy())
                example_images.append(wandb.Image(data[0]))
                error = max(error, ((output - target) / target).max())
            else:
                error = np.inf
    return val_loss, example_images, error
