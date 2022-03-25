from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from tqdm import tqdm

from . import inspect_code


class ScriptCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, dirpath="./", datamodule=None):
        self.dirpath = Path(dirpath)
        self.datamodule = datamodule

    def on_train_begin(self, logs=None):
        self.dirpath.mkdir(parents=True, exist_ok=True)
        with open(Path(self.dirpath) / "model_script.txt", "w") as file:
            model_script = inspect_code.get_class_code(type(self.model))
            file.write(model_script)
        with open(Path(self.dirpath) / "generator_script.txt", "w") as file:
            generator_script = inspect_code.get_class_code(type(self.model.generator))
            file.write(generator_script)
        with open(Path(self.dirpath) / "discriminator_script.txt", "w") as file:
            discriminator_script = inspect_code.get_class_code(
                type(self.model.discriminator)
            )
            file.write(discriminator_script)
        if self.datamodule is not None:
            filename_datamodule = Path(self.dirpath) / "datamodule_script.txt"
            with open(filename_datamodule, "w") as file:
                file.write(inspect_code.get_class_code(type(self.datamodule)))
