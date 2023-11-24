import sys

from logging import getLogger

from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

logger = getLogger(name=__name__)


def train(model, train_data, val_data=None):
    fit_log = StringIO()
    logger.info(msg=f"Training ...")

    with redirect_stdout(new_target=fit_log), redirect_stderr(
            new_target=fit_log):
        model.fit(train_data, log_cout=sys.stdout, log_cerr=sys.stderr)
        if val_data != None: 
            y = pd.DataFrame(val_data.data, columns=['label'])
            loss = validate(model, val_data.pack_to_catboost(), y)
        else: loss = model.get_best_score()

    logger.info(msg=f"Training log: \n{fit_log.getvalue()}")
    return model, loss

    
def validate(model, x, y):
    logger.info(msg=f"Validating ..")

    unlabeled_data = x

    prediction = model.predict(data=unlabeled_data)
    
    f1 = f1_score(y, prediction)
    logger.info(msg=f"Validate finished.")
    return f1



def inference(model, data, keys, save_dir):
    logger.info(msg=f"Inferencing ...")

    # model = parameters["model"]
    unlabeled_data = data

    prediction = model.predict(data=unlabeled_data)

    df = keys
    df = pd.concat(objs=[df, pd.Series(data=prediction, name="label")], axis=1)

    df.to_csv(path_or_buf=save_dir, index=False)

    logger.info(msg=f"Number 1s: {np.count_nonzero(a = prediction == 1)}")
