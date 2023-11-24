from copy import copy
import numpy as np
import pandas as pd
import random
import os
import logging
from configparser import ConfigParser
logger = logging.getLogger(__name__)
def time2seconds(time):
    time = str(time)
    if time == '': return 0
    t = copy(time)
    while len(t) < 6:
        t = '0' + t
    sec = int(copy(t[-2:]))
    t = t[:-2]
    min = int(copy(t[-2:]))
    t = t[:-2]
    hour = int(copy(t))
    return (sec+min*60+hour*60*60)//60

def init_seeds(npSeed = 0, pdSeed = 0, randomSeed = 0):
    np.random.seed(seed=npSeed)
    pd.core.common.random_state(pdSeed)
    random.seed(a=randomSeed)

    logger.info(msg=f"Seeds: \n Numpy: {npSeed}\n Pandas: {pdSeed}\n Random: {randomSeed}")


def parse_configs(configs_path: str) -> ConfigParser:
    parser = ConfigParser()
    parser.read(filenames=configs_path)

    return parser

def get_version(root):
    file_list = os.listdir(root+'checkpoint/')
    counter = 1
    while f'checkpoint_{counter}' in file_list:
        counter += 1
    return counter


def get_logger(log_name: str, save_dir: str) -> logging.Logger:
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(level=logging.DEBUG)
    sh.setFormatter(fmt=formatter)

    filename = os.path.join(save_dir, log_name)
    if os.path.exists(filename):
        os.remove(filename)
    fh = logging.FileHandler(filename=filename, mode="a", encoding="UTF-8")
    fh.setLevel(level=logging.DEBUG)
    fh.setFormatter(fmt=formatter)

    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    logger.addHandler(hdlr=sh)
    logger.addHandler(hdlr=fh)

    return logger
