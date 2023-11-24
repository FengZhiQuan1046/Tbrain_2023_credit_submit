from tools import *
import logging
from trainer import *
from data import tbrain_data
from model import *
import os
configs_path = './configs/configs.ini'

configs = parse_configs(configs_path=configs_path)

output_root = configs.get(section="GENERAL", option="output_dir")
version = 4#get_version(output_root)
logging_name = f"logging_{version}.log"

model_configs = {
    'model_name' : configs.get(section="MODEL", option="model_name"),
    'checkpoint_version' : 'None'
}

logger = get_logger(logging_name, output_root+'logging/')
# logger.setLevel(logging.INFO)
init_seeds()
model = initialize_model(model_configs)

data = tbrain_data(split='trainval')
# ddd = data.data.label
train(model, data.pack_to_catboost())

import pickle as pkl
with open(f'{output_root}checkpoint/mod{version}.pkl', 'wb') as f:
    pkl.dump(model, f)
    
data = tbrain_data('test')
keys = data.data.txkey.copy()
inference(mod, data.pack_to_catboost(), keys, f"{output_root}prediction/pred{version}.csv")

data = tbrain_data('val_previous')
keys = data.data.txkey.copy()
inference(mod, data.pack_to_catboost(), keys, f"{output_root}prediction/pred{version}_val.csv")


import pandas as pd

val_pred = pd.read_csv(f"{output_root}prediction/pred{version}_val.csv")
test_pred = pd.read_csv(f"{output_root}prediction/pred{version}.csv")

integrate = pd.concat([val_pred, test_pred])
count = [0,0, 0]
for each in list(integrate.label):
    if each == 0:
        count[0]+=1
    elif each ==1:
        count[1]+=1
    else:
        count[2]+=1
integrate.to_csv(path_or_buf=f'{output_root}prediction.csv', index=False)
