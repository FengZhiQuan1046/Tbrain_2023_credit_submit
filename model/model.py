import os
import random

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool
from configparser import ConfigParser
from logging import getLogger
import json

logger = getLogger(__name__)


def initialize_model(configs: dict, model_params=None):
    models = {"CatBoostClassifier": CatBoostClassifier}

    model_name = configs["model_name"]
    if model_params == None:
        model_params_path = os.path.join("configs", (model_name + ".json"))
        with open(file=model_params_path, mode="r", encoding="UTF-8") as file:
            model_params = json.load(fp=file)
    model = models[model_name](**model_params)


    logger.info(msg=f"The details of {model_name}: \n{model.get_params()}")
    logger.info(msg=f"{model_name} has been initialized.")

    return model