import os
import pickle
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import wandb
from sklearn.preprocessing import MinMaxScaler


def login():
    """Logs a session in wandb.

    Returns:
        pathlib.Path: path of the repository in the cloud. The path depends whether we are on Colab or Kaggle.
    """
    IS_COLAB = "google.colab" in sys.modules
    IS_KAGGLE = "kaggle_secrets" in sys.modules
    if IS_KAGGLE:
        from kaggle_secrets import UserSecretsClient

        WANDB_API = UserSecretsClient().get_secret("wandb_api")
    elif IS_COLAB:
        WANDB_API = "3e384d0e21fd4f06a6abc2fdc162b88eadc00994"
    else:
        WANDB_API = os.getenv("WANDB_API")
    wandb.login(key=WANDB_API)
