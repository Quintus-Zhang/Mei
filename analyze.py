import os
import numpy as np
import pandas as pd
import glob
import warnings

# keras items
from keras.models import load_model

# local items
from scan import Scan

# read the 'Mei_NN_*.csv' file
# load weights of the best model based on a specific metric
# - return the model with lowest loss_val
# - return the model with lowest Diff_#_D60_val
# use which test dataset
# - that from current 1m dataset
# - that from extra source


def select_and_predict(round_no=None, based_on='Loss_val', extra_source=False):
    """ Select best model based on the given metric, and predict on the given test dataset

    :param round_no:
    :param based_on:
    :param extra_source:
    :return:
    """

    if round_no is None:
        warnings.warn('round_no has to be specified')
        return

    # set up paths of directories and files
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, 'Results')
    round_dir = os.path.join(results_dir, f'Mei_NN_{round_no}')
    round_fp = os.path.join(round_dir, f'Mei_NN_{round_no}.csv')
    temp_dir = os.path.join(base_dir, 'Temp')

    # read the 'Mei_NN_*.csv' file
    res = pd.read_csv(round_fp)

    # search for file name of the best model based on a specific metric
    if based_on in ['Loss_val', 'Diff_#_D60_val']:
        params_idx = res.loc[:, based_on].values.argmin()
    elif based_on in ['PR_AUC_val', 'ROC_AUC_val', 'F_score_val', 'G_mean_val']:
        params_idx = res.loc[:, based_on].values.argmax()
    else:
        warnings.warn(f'Cannot select the best model based on {based_on}')
    best_model_fp = glob.glob(f'{round_dir}\\{params_idx}_*')[0]
    best_model = load_model(best_model_fp)

    # retrieve test dataset
    if not extra_source:
        X_test = np.load(f'{temp_dir}\\X_test.pkl')
        y_test = np.load(f'{temp_dir}\\y_test.pkl')
    else:
        warnings.warn(f'Test dataset from extra source is currently not supported.')

    # predict
    metrics_test = Scan.model_predict(best_model, X_test, y_test)
    return metrics_test


if __name__ == "__main__":
    metrics_test = select_and_predict(round_no=3, based_on='Diff_#_D60_val', extra_source=False)
    print(metrics_test)


