import os
import numpy as np
import pandas as pd
import glob
import warnings

# keras items
from keras.models import load_model

# local items
from scan import Scan
from config import results_dir, temp_dir, data_fp, os_data_fp
from utils import DataPrep, DataPrepWrapper

# read the 'Mei_NN_*.csv' file
# load weights of the best model based on a specific metric
# - return the model with lowest loss_val
# - return the model with lowest Diff_#_D60_val
# use which test dataset
# - that from current 1m dataset
# - that from extra source


def select_best_model(round_no=None, based_on='Loss_val', index=None, best_k=1):
    """ Select best model based on the given metric

    :param round_no:
    :param based_on:
    :return:
    """

    if round_no is None:
        warnings.warn('round_no has to be specified')
        return

    round_dir = os.path.join(results_dir, f'Mei_NN_{round_no}')
    round_fp = os.path.join(round_dir, f'Mei_NN_{round_no}.csv')

    if based_on is 'index':
        return load_model(glob.glob(f'{round_dir}\\{index}_*')[0])

    # read the 'Mei_NN_*.csv' file
    res = pd.read_csv(round_fp)

    # search for file name of the best model based on a specific metric
    if based_on in ['Loss_val', 'Diff_D60_val']:
        params_idx = res.loc[:, based_on].sort_values().index[:best_k]
    elif based_on in ['PR_AUC_val', 'ROC_AUC_val', 'F_score_val', 'G_mean_val']:
        params_idx = res.loc[:, based_on].sort_values(ascending=False).index[:best_k]
    else:
        warnings.warn(f'Cannot select the best model based on {based_on}')

    models = []
    for idx in params_idx:
        best_model_fp = glob.glob(f'{round_dir}\\{idx}_*')[0]
        best_model = load_model(best_model_fp)
        models.append(best_model)
    return models


def predict_on_test(best_model):
    # retrieve test dataset
    X_test = np.load(f'{temp_dir}\\X_test.pkl')
    y_test = np.load(f'{temp_dir}\\y_test.pkl')

    # predict
    metrics_test = Scan.model_predict(best_model, X_test, y_test)
    return metrics_test


if __name__ == "__main__":
    # get model
    models = select_best_model(round_no=6, based_on='Loss_val', index=None, best_k=10)

    # predict
    for model in models:
        metrics_res = predict_on_test(model)
        # compare
        print(metrics_res)


# round_3
# not good, hold-out validation is not enough

