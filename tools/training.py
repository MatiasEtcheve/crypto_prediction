from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score

from tools import inspect_code


class ScriptCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, dirpath="./", datamodule=None):
        self.dirpath = Path(dirpath)
        self.datamodule = datamodule

    def on_train_begin(self, logs=None):
        self.dirpath.mkdir(parents=True, exist_ok=True)
        with open(Path(self.dirpath) / "model_script.txt", "w") as file:
            model_script = inspect_code.get_class_code(type(self.model))
            file.write(model_script)

        for attr in ["generator", "discriminator"]:
            if hasattr(self.model, attr):
                with open(Path(self.dirpath) / f"{attr}_script.txt", "w") as file:
                    generator_script = inspect_code.get_class_code(
                        type(getattr(self.model, attr))
                    )
                    file.write(generator_script)

        if self.datamodule is not None:
            filename_datamodule = Path(self.dirpath) / "datamodule_script.txt"
            with open(filename_datamodule, "w") as file:
                file.write(inspect_code.get_class_code(type(self.datamodule)))
