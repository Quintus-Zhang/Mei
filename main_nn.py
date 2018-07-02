# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

import os
import time
import numpy as np
import pandas as pd
import pickle as pk
import multiprocessing

# local items
from utils import data_prep
from scan import Scan
from params import ParamsGridSearch, ParamsRandomSearch
from config import *


# TODO: Add Sphinx documentation
# TODO: Add visualization (search keras hps tuning)
if __name__ == "__main__":
    start = time.time()
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, 'Results')
    data_fp = os.path.join(base_dir, 'Data', 'Gse_panel_current_sample_raw.csv')
    temp_dir = os.path.join(base_dir, 'Temp')

    # retrieve training set, validation set, and test set
    df = pd.read_csv(data_fp)
    X_train, X_val, X_test, y_train, y_val, y_test = data_prep(df)

    # dump the test dataset as pickle file to Temp dir for later use
    X_test.dump(f'{temp_dir}\\X_test.pkl')
    y_test.dump(f'{temp_dir}\\y_test.pkl')

    prs = ParamsRandomSearch(**params)
    print(f'# of combos: {len(prs.params_grid)}')

    multiprocessing.freeze_support()
    p = multiprocessing.Process()
    p.start()

    # scan the params grid
    t = Scan(X_train=X_train,
             y_train=y_train,
             X_val=X_val,
             y_val=y_val,
             params_grid=prs.params_grid,
             dataset_name='Mei_NN',
             model=neural_nets)

    p.terminate()
    end = time.time()
    print(end-start)
