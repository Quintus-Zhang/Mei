# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

# params are exactly the same, but metrics results are little bit different

import os
import time
import glob
import numpy as np
import pandas as pd
import pickle as pk
import multiprocessing

# local items
from utils import data_prep, DataPrep, DataPrepWrapper
from scan import Scan
from params import ParamsGridSearch, ParamsRandomGridSearch, ParamsRandomSearch
from config import *


# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO: Add Sphinx documentation
# TODO: Add visualization (search keras hps tuning)
if __name__ == "__main__":
    start = time.time()

    # # retrieve training set, validation set, and test set
    # # 1. in sample data only
    # df = pd.read_csv(data_fp)
    # X_train, X_val, X_test, y_train, y_val, y_test = data_prep(df)

    # 2. DataPrep
    # df = pd.read_csv(data_fp)
    # is_data = DataPrep(df)
    # X_train, X_val, X_test, y_train, y_val, y_test = is_data.split_and_standardize()

    # # 3. DataPrepWrapper
    df = pd.read_csv(data_fp)
    os_df = pd.read_csv(os_data_fp)
    is_data = DataPrep(df)
    data = DataPrepWrapper(is_data, os_df)
    X_train, X_val, X_test, y_train, y_val, y_test = data.split_and_standardize()

    # dump the test dataset as pickle file to Temp dir for later use
    if not glob.glob(f'{temp_dir}\\*.pkl'):
        X_test.dump(f'{temp_dir}\\X_test.pkl')
        y_test.dump(f'{temp_dir}\\y_test.pkl')

    prs = ParamsRandomSearch(params, n_iter=n_iter)
    # print(f'# of combos: {len(prs.params_grid)}')
    # print(prs.params_grid)

    multiprocessing.freeze_support()
    p = multiprocessing.Process()
    p.start()

    # scan the params grid
    t = Scan(X_train=X_train,
             y_train=y_train,
             X_val=X_val,
             y_val=y_val,
             params_search=prs,
             dataset_name='Mei_NN',
             model=neural_nets)

    p.terminate()
    end = time.time()
    print(end-start)
