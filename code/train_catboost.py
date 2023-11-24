from tools import *
import logging
from trainer import *
from data import tbrain_data
from model import *
import os
import optuna
import pickle as pkl
configs_path = './configs/configs.ini'

configs = parse_configs(configs_path=configs_path)

output_root = configs.get(section="GENERAL", option="output_dir")
version = configs.get(section="GENERAL", option="version")#get_version(output_root)
logging_name = f"logging_{version}.log"

model_configs = {
    'model_name' : configs.get(section="MODEL", option="model_name")
    # 'checkpoint_version' : 'None'
}

configs_save_dir = configs.get(section="GENERAL", option="configs_dir")
# model_params = get_model_parameters(model_params_path=configs.get(section="MODEL", option="config_path"))

logger = get_logger(logging_name, output_root+'logging/')
# logger.setLevel(logging.INFO)
init_seeds()


data = tbrain_data('train', preprocess=True).pack_to_catboost()
train_features_weight = False
train_regular_param = True

max_f1 = 0.
configs_counter = 0
trial_counter = 0
val_data = tbrain_data('val', preprocess=True)

with open('./configs/CatBoostClassifier.json', 'r') as f:
    model_params = json.load(f)
def opt(trial):
    global train_features_weight, train_regular_param, max_f1, configs_counter, trial_counter
    if train_regular_param:
        optim_params = {
            "depth": trial.suggest_int("depth", 5, 10, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 5.0, log=True),
            "learning_rate": trial.suggest_float('lr', 0.01, 1.2, log=True),
            "class_weights": {0: trial.suggest_float("label_w0", 0.5, 2.0, log=True), 
                              1: trial.suggest_float("label_w1", 0.8, 4.0, log=True)}
        }
        for k in optim_params:
            model_params[k] = optim_params[k]
    if train_features_weight:
        model_params["feature_weights"] = {
                        "txkey":0,
                        "locdt":1.,
                        "loctm":trial.suggest_float("w_loctm", 0.01, 10.0, log=True),
                        "chid":trial.suggest_float("w_chid", 0.01, 10.0, log=True),
                        "cano":trial.suggest_float("w_cano", 0.01, 10.0, log=True),
                        "contp":trial.suggest_float("w_contp", 0.01, 10.0, log=True),
                        "etymd":trial.suggest_float("w_etymd", 0.01, 10.0, log=True),
                        "mchno":trial.suggest_float("w_mchno", 0.01, 10.0, log=True),
                        "acqic":trial.suggest_float("w_acqic", 0.01, 10.0, log=True),
                        "mcc":trial.suggest_float("w_mcc", 0.01, 10.0, log=True),
                        "conam":trial.suggest_float("w_conam", 0.01, 10.0, log=True),
                        "ecfg":trial.suggest_float("w_ecfg", 0.01, 10.0, log=True),
                        "insfg":trial.suggest_float("w_insfg", 0.01, 10.0, log=True),
                        "iterm":trial.suggest_float("w_iterm", 0.01, 10.0, log=True),
                        "bnsfg":trial.suggest_float("w_bnsfg", 0.01, 10.0, log=True),	
                        "flam1":trial.suggest_float("w_flam1", 0.01, 10.0, log=True),
                        "stocn":trial.suggest_float("w_stocn", 0.01, 10.0, log=True),
                        "scity":trial.suggest_float("w_scity", 0.01, 10.0, log=True),
                        "stscd":trial.suggest_float("w_stscd", 0.01, 10.0, log=True),
                        "ovrlt":trial.suggest_float("w_ovrlt", 0.01, 10.0, log=True),
                        "flbmk":trial.suggest_float("w_flbmk", 0.01, 10.0, log=True),
                        "hcefg":trial.suggest_float("w_hcefg", 0.01, 10.0, log=True),
                        "csmcu":trial.suggest_float("w_csmcu", 0.01, 10.0, log=True),
                        "csmam":trial.suggest_float("w_csmam", 0.01, 10.0, log=True),
                        "flg_3dsmk":trial.suggest_float("w_flg_3dsmk", 0.01, 10.0, log=True)
                    }

    model = initialize_model(configs=model_configs, model_params=model_params)
    model, loss = train(model, data, val_data)
    trial_counter += 1
    # logger.info(msg=f"================================>>>>>  loss: {loss} <<<<<================================")
    # print(loss)
    f1 = loss#['learn']['F1:use_weights=false']
    if max_f1 < f1 or f1 >= 0.998:
        if max_f1 < f1: max_f1 = f1
        config_2b_saved = copy(model_params)
        config_2b_saved['f1'] = f1
        with open(os.path.join(configs_save_dir, f'config_{trial_counter}_{configs_counter}.json'), 'w') as f:
            json.dump(config_2b_saved, f)
        configs_counter += 1
        with open(f"{output_root}checkpoint/{version}/{configs_counter}.mod", 'wb') as f:
            pkl.dump(model, f)
    return f1
study = optuna.create_study(direction="maximize", study_name="autoctb")
study.optimize(opt, n_trials=100)
best_trial = study.best_trial()


import pickle as pkl
with open(output_root+"medium/best_trial.pkl", 'wb') as f:
    pkl.dump(best_trial, f)