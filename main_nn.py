from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import os
import time
import pickle as pk
import pandas as pd
import numpy as np
import multiprocessing

# Keras items
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import Adam, Nadam
from keras.activations import relu, elu, sigmoid
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

import tensorflow as tf

from utils import data_prep, ModelCheckpointRtnBest
from scan import Scan
from params import ParamsGridSearch, ParamsRandomSearch
from layers import add_other_hidden_layers


# Set up your model
def neural_nets(X_train, y_train, X_val, y_val, params, params_idx, cp_dir):
    """

    :param X_train:
    :param y_train:
    :param X_val:
    :param y_val:
    :param params:
    :param params_idx:
    :param cp_dir: output directory of the checkpoints(weights)
    :return:
    """
    # TODO: do not return history
    # TODO: try to return the index of epoch where model training is stopped
    # TODO: add file path for the hd5f files saving weights
    # TODO:
    # sess = tf.Session()
    # K.set_session(sess)

    model = Sequential()
    model.add(Dense(params['layer_size'],
                    input_dim=X_train.shape[1],
                    activation=params['activation'],
                    kernel_initializer=params['kernel_initializer']))

    model.add(Dropout(params['dropout']))

    add_other_hidden_layers(model, params, 1)

    model.add(Dense(1, activation=params['last_activation'],
                    kernel_initializer=params['kernel_initializer']))

    model.compile(loss=params['losses'],
                  optimizer=params['optimizer']())

    # set up callbacks
    cp_fp = f'{cp_dir}\\{params_idx}' + '_best_model_{epoch:02d}_{val_loss:.5f}.hdf5'
    check_pointer = ModelCheckpointRtnBest(filepath=cp_fp, monitor='val_loss', mode='min', verbose=1)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
    cb_list = [check_pointer, early_stopper]

    model.fit(X_train, y_train,
              validation_data=[X_val, y_val],
              batch_size=params['batch_size'],
              callbacks=cb_list,                    # , TQDMNotebookCallback() PlotLossesKeras()
              epochs=params['epochs'],
              verbose=0)
    print(model.summary())
    return model


# TODO: Add Sphinx documentation
if __name__ == "__main__":
    start = time.time()
    # Get training set, validation set, and test set
    base_path = os.path.dirname(__file__)
    results_path = os.path.join(base_path, 'Results')
    data_fp = os.path.join(base_path, 'Data', 'Gse_panel_current_sample_raw.csv')

    df = pd.read_csv(data_fp)
    X_train, X_val, X_test, y_train, y_val, y_test = data_prep(df)

    # params grid
#    params = {'lr': [10**i for i in range(-6, 1)],
#              'first_neuron': list(range(4, 46, 10)),
#               'batch_size': [2**i for i in range(2, 13, 2)],
#               'epochs': list(range(5, 100, 20)),
#               'dropout': [0],
#               'kernel_initializer': ['uniform', 'normal'],
#               'optimizer': [Adam, Nadam],
#               'losses': [binary_crossentropy],
#               'activation': [relu],
#               'last_activation': [sigmoid]}

    # params_rs = {'lr': [1.00184520454465e-6],
    #           'dropout': [0.360162246721079],
    #
    #           'batch_size': [335],
    #           'epochs': [100],
    #
    #           'layer_size': [147],
    #           'other_hidden_layers': [0],
    #           'shapes': ['funnel'],
    #
    #           'kernel_initializer': ['normal'],
    #           'optimizer': [Adam],
    #           'losses': [binary_crossentropy],
    #           'activation': [relu],
    #           'last_activation': [sigmoid]}

    params_rs = {'lr': [0.110222803767004],
              'dropout': [0.208511002351287],

              'batch_size': [1161],
              'epochs': [100],

              'layer_size': [147],
              'other_hidden_layers': [2],
              'shapes': ['funnel'],

              'kernel_initializer': ['normal'],
              'optimizer': [Adam],
              'losses': [binary_crossentropy],
              'activation': [relu],
              'last_activation': [sigmoid]}

    # params_rs = {'lr': (-6, 1, 5),   # log scale for lr
    #              'dropout': (0, 0.5, 2),
    #
    #              'batch_size': (100, 2000, 5),
    #              'epochs': [100],
    #
    #              'layer_size': (10, 200, 5),
    #              'other_hidden_layers': [0, 1, 2, 3],
    #              'shapes': ['funnel'],
    #
    #              'kernel_initializer': ['normal'],
    #              'optimizer': [Adam],
    #              'losses': [binary_crossentropy],
    #              'activation': [relu],
    #              'last_activation': [sigmoid]
    #              }

    prs = ParamsRandomSearch(**params_rs)
    print(f'# of combos: {len(prs.params_grid)}')

    multiprocessing.freeze_support()
    p = multiprocessing.Process()
    p.start()

    # scan the params grid
    t = Scan(X_train=X_train,
             y_train=y_train,
             X_val=X_test,
             y_val=y_test,
             params_grid=prs.params_grid,
             dataset_name='Mei_NN',
             model=neural_nets)

    p.terminate()
    end = time.time()
    print(end-start)
