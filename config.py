import os
import pandas as pd
import matplotlib.pyplot as plt

# xgboost
import xgboost as xgb

# Keras items
from keras.models import Sequential
from keras.layers import Dropout, Dense, LeakyReLU
from keras.optimizers import Adam, Nadam
from keras.activations import relu, elu, sigmoid
from keras.losses import binary_crossentropy, logcosh
from keras.callbacks import EarlyStopping
from keras.initializers import he_normal, he_uniform, random_normal, random_uniform

# tensorflow items
from keras import backend as K
import tensorflow as tf
import tensorflow as tf

# local items
from utils import ModelCheckpointRtnBest
from layers import add_other_hidden_layers

# set up paths of directories and files
base_dir = os.path.dirname(__file__)
results_dir = os.path.join(base_dir, 'Results')
temp_dir = os.path.join(base_dir, 'Temp')
data_dir = os.path.join(base_dir, 'Data')
data_fp = os.path.join(data_dir, 'Gse_panel_current_sample_raw.csv')
os_data_fp = os.path.join(data_dir, 'Gse_2016_ltvgt80_v50.csv')

X_cols=['orig_rt', 'orig_upb', 'oltv', 'num_bo', 'dti', 'num_unit', 'fico',
       'oyr', 'oqtr', 'ind_ede', 'pmms_o', 'avg_upb', 'OUPB_Rel',
       'loan_age_qtr', 'year', 'qtr', 'PMMS', 'HPI_O', 'HPI', 'ur',
       'CUPB_calc', 'Orig_value', 'CLTV', 'orig_chn_B', 'orig_chn_C',
       'orig_chn_R', 'orig_chn_T', 'loan_purp_C', 'loan_purp_N', 'loan_purp_P',
       'loan_purp_R', 'loan_purp_U', 'prop_type_CO', 'prop_type_CP',
       'prop_type_LH', 'prop_type_MH', 'prop_type_PU', 'prop_type_SF',
       'occ_stat_I', 'occ_stat_O', 'occ_stat_P', 'occ_stat_S', 'judicial_st_N',
       'judicial_st_Y', 'fhb_flag_N', 'fhb_flag_Y']


#########################################
#          Set up your model            #
#########################################
def neural_nets(X_train, y_train, X_val, y_val, params, params_idx, cp_dir):
    model = Sequential()
    if type(params['activation']) is not type:
        model.add(Dense(params['layer_size'],
                        input_dim=X_train.shape[1],
                        activation=params['activation'],
                        kernel_initializer=params['kernel_initializer']))
    else:
        model.add(Dense(params['layer_size'],
                        input_dim=X_train.shape[1],
                        kernel_initializer=params['kernel_initializer']))
        model.add(params['activation']())   # TODO: look into this issue

    model.add(Dropout(params['dropout']))

    add_other_hidden_layers(model, params, 1)

    model.add(Dense(1, activation=params['last_activation'],
                    kernel_initializer=params['kernel_initializer']))

    model.compile(loss=params['losses'],
                  optimizer=params['optimizer'](lr=params['lr']))

    # set up callbacks
    cp_fp = f'{cp_dir}\\{params_idx}' + '_best_model_{epoch:02d}_{val_loss:.5f}.hdf5'
    check_pointer = ModelCheckpointRtnBest(filepath=cp_fp, monitor='val_loss', mode='min', verbose=0)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=5, verbose=0, mode='min')
    cb_list = [check_pointer, early_stopper]

    model.fit(X_train, y_train,
              validation_data=[X_val, y_val],
              batch_size=params['batch_size'],
              callbacks=cb_list,                    # , TQDMNotebookCallback() PlotLossesKeras()
              epochs=params['epochs'],
              verbose=0,
              class_weight={0: 1, 1: params['pos_weight']})
    # print(model.summary())
    return model, early_stopper.stopped_epoch


#########################################
#        Set Params Search Range        #
#########################################

# (start, end, # of points)
# [first item, second item, ...]

# params grid
# params = {'lr': [10**i for i in range(-6, 1)],
#              'first_neuron': list(range(4, 46, 10)),
#               'batch_size': [2**i for i in range(2, 13, 2)],
#               'epochs': list(range(5, 100, 20)),
#               'dropout': [0],
#               'kernel_initializer': ['uniform', 'normal'],
#               'optimizer': [Adam, Nadam],
#               'losses': [binary_crossentropy],
#               'activation': [relu],
#               'last_activation': [sigmoid]}

# params = {'lr': [1.00184520454465e-6],
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

# params = {'lr': [0.110222803767004],
#           'dropout': [0.208511002351287],
#
#           'batch_size': [1161],
#           'epochs': [100],
#
#           'layer_size': [147],
#           'other_hidden_layers': [2],
#           'shapes': ['funnel'],
#
#           'kernel_initializer': ['normal'],
#           'optimizer': [Adam],
#           'losses': [binary_crossentropy],
#           'activation': [relu],
#           'last_activation': [sigmoid]}


# params = {'lr': [0.001],   # log scale for lr
#           'dropout': [0.2],
#
#           'batch_size': (100, 2000, 5),
#           'epochs': [100],
#
#           'layer_size': [20],
#           'other_hidden_layers': [1],
#           'shapes': ['funnel'],
#
#           'kernel_initializer': ['normal'],
#           'optimizer': [Adam],
#           'losses': [binary_crossentropy],
#           'activation': [relu],
#           'last_activation': [sigmoid]
#          }

# params = {'lr': (-6, 1, 5),   # log scale for lr
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

# 183665 min = 51 hours
# params = {'lr': (-7, 1),                    # log scale for lr
#           'dropout': (0, 0.5),
#
#           'batch_size': (100, 3000),
#           'epochs': [100],
#
#           'layer_size': (10, 300),
#           'other_hidden_layers': [0, 1, 2, 3, 4],
#           'shapes': ['funnel', 'rhombus', 'long_funnel',
#                      'hexagon', 'triangle', 'stairs'],
#
#           'kernel_initializer': ['normal', 'uniform'],
#           'optimizer': [Adam],
#           'losses': [binary_crossentropy],
#           'activation': [relu],
#           'last_activation': [sigmoid]
#           }

# # 35195 seconds
# params = {'lr': (-6, 0),                    # log scale for lr
#           'dropout': (0, 0.4),
#
#           'batch_size': (1000, 4000),
#           'epochs': [100],
#
#           'layer_size': (10, 200),
#           'other_hidden_layers': [0, 1, 2],
#           'shapes': ['funnel', 'long_funnel', 'hexagon'],
#
#           'kernel_initializer': ['uniform'],
#           'optimizer': [Adam, Nadam],
#           'losses': [binary_crossentropy, logcosh],
#           'activation': [relu, elu, LeakyReLU],
#           'last_activation': [sigmoid]
#           }

# # MEI_NN_5
# params = {'lr': (-6, -1),                    # log scale for lr
#           'dropout': (0, 0.4),
#
#           'batch_size': (10, 2000),
#           'epochs': [100],
#
#           'layer_size': (10, 400),
#           'other_hidden_layers': [0, 1],
#           'shapes': ['funnel'],
#
#           'pos_weight': (5, 500),
#
#           'kernel_initializer': ['uniform'],
#           'optimizer': [Adam, Nadam, RMSprop],
#           'losses': [binary_crossentropy],
#           'activation': [relu, elu, LeakyReLU],
#           'last_activation': [sigmoid],
#           }


# 73876 seconds
# params = {'lr': (-6, -1),                    # log scale for lr
#           'dropout': (0, 0.4),
#
#           'batch_size': (10, 2000),
#           'epochs': [100],
#
#           'layer_size': (10, 500),
#           'other_hidden_layers': [0, 1],
#           'shapes': ['funnel'],
#
#           'pos_weight': [1],
#
#           'kernel_initializer': ['uniform'],
#           'optimizer': [Adam, Nadam, RMSprop],
#           'losses': [binary_crossentropy],
#           'activation': [relu, elu, LeakyReLU],
#           'last_activation': [sigmoid],
#           }

# # Mei_NN_7
# params = {'lr': (-6, -1),                    # log scale for lr
#           'dropout': (0, 0.5),
#
#           'batch_size': (10, 2000),
#           'epochs': [100],
#
#           'layer_size': (10, 500),
#           'other_hidden_layers': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
#           'shapes': ['funnel'],
#
#           'pos_weight': [1],
#
#           'kernel_initializer': ['uniform'],
#           'optimizer': [Adam, Nadam, RMSprop],
#           'losses': [binary_crossentropy],
#           'activation': [relu, elu, LeakyReLU],
#           'last_activation': [sigmoid],
#           }

# Mei_NN_9 and Mei_NN_10
# params = {'lr': (-6, -1),
#           'dropout': (0, 0.5),
#
#           'batch_size': (10, 2000),
#           'epochs': [100],
#
#           'layer_size': (10, 500),
#           'other_hidden_layers': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
#           'shapes': ['funnel'],
#
#           'pos_weight': [1],
#
#           'kernel_initializer': ['uniform'],
#           'optimizer': [Adam, Nadam, RMSprop],
#           'losses': [binary_crossentropy, l1, l2],
#           'activation': [relu, elu, LeakyReLU],
#           'last_activation': [sigmoid],
#           }

# # Mei_NN_11,  new data
# params = {'lr': (-6, -1),
#           'dropout': (0, 0.5),
#
#           'batch_size': (10, 2000),
#           'epochs': [100],
#
#           'layer_size': (10, 500),
#           'other_hidden_layers': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
#           'shapes': ['funnel'],
#
#           'pos_weight': [1],
#
#           'kernel_initializer': ['uniform'],
#           'optimizer': [Adam, Nadam, RMSprop],
#           'losses': [binary_crossentropy, l1, l2],
#           'activation': [relu, elu, LeakyReLU],
#           'last_activation': [sigmoid],
#           }
#
# # Mei_NN_12,  new test data, one hidden layer (benchmark)
# params = {'lr': (-6, -1),
#           'dropout': (0, 0.7),
#
#           'batch_size': (10, 2000),
#           'epochs': [100],
#
#           'layer_size': (10, 500),
#           'other_hidden_layers': [0],
#           'shapes': ['funnel'],
#
#           'pos_weight': [1],
#
#           'kernel_initializer': ['uniform'],
#           'optimizer': [Adam, Nadam, RMSprop],
#           'losses': [binary_crossentropy],
#           'activation': [relu, elu, LeakyReLU],
#           'last_activation': [sigmoid],
#           }
#
#
# # Mei_NN_13,  new test data
# params = {'lr': (-6, -1),
#           'dropout': (0, 0.7),
#
#           'batch_size': (200, 2000),
#           'epochs': [100],
#
#           'layer_size': (40, 500),
#           'other_hidden_layers': [0, 1, 3, 5, 7, 9],
#           'shapes': ['funnel'],
#
#           'pos_weight': [1],
#
#           'kernel_initializer': ['uniform'],
#           'optimizer': [Adam, Nadam, RMSprop],
#           'losses': [binary_crossentropy],
#           'activation': [relu, elu, LeakyReLU],
#           'last_activation': [sigmoid],
#           }

# # Mei_NN_14,  new test data
# # # see if nn with 6 hidden layers can beat the nn with 1 hidden layer ?
# # params = {'lr': (-6, -1),
# #           'dropout': (0, 0.7),
# #
# #           'batch_size': (500, 2500),
# #           'epochs': [100],
# #
# #           'layer_size': (40, 500),
# #           'other_hidden_layers': [5],
# #           'shapes': ['funnel'],
# #
# #           'pos_weight': [1],
# #
# #           'kernel_initializer': ['uniform'],
# #           'optimizer': [Adam, Nadam, RMSprop],
# #           'losses': [binary_crossentropy],
# #           'activation': [relu, elu, LeakyReLU],
# #           'last_activation': [sigmoid],
# #           }


# # Mei_NN_16,  new test data
# # try cloglog function in the last layer, compare the results with Mei_NN_12
# params = {'lr': (-6, -1),
#           'dropout': (0, 0.7),
#
#           'batch_size': (10, 2000),
#           'epochs': [100],
#
#           'layer_size': (10, 500),
#           'other_hidden_layers': [0],
#           'shapes': ['funnel'],
#
#           'pos_weight': [1],
#
#           'kernel_initializer': ['uniform'],
#           'optimizer': [Adam, Nadam, RMSprop],
#           'losses': [binary_crossentropy],
#           'activation': [relu, elu, LeakyReLU],
#           'last_activation': [cloglog],
#           }

# # Mei_NN_19 and 20,  gse test data
# # try early stopping from 1e-5 to 1e-6
# # try other initializers
# # try higher lb of layer_size
# # try 1 hidden layer and 2 hidden layers
# params = {'lr': (-6, -1),
#           'dropout': (0, 0.7),
#
#           'batch_size': (100, 2000),
#           'epochs': [100],
#
#           'layer_size': (100, 300),
#           'other_hidden_layers': [0, 1],
#           'shapes': ['funnel'],
#
#           'pos_weight': [1],
#
#           'kernel_initializer': ['normal', 'uniform', 'he_normal', 'he_uniform', 'glorot_uniform'],
#           'optimizer': [Adam, Nadam, RMSprop],
#           'losses': [binary_crossentropy],
#           'activation': [relu, elu, LeakyReLU],
#           'last_activation': [sigmoid],
#           }
#

# # Mei 21
# # large weight
# # 86084
# params = {'lr': (-6, -1),
#           'dropout': (0, 0.7),
#
#           'batch_size': (100, 2000),
#           'epochs': [100],
#
#           'layer_size': (100, 300),
#           'other_hidden_layers': [0, 1],
#           'shapes': ['funnel'],
#
#           'pos_weight': [500],
#
#           'kernel_initializer': ['normal', 'uniform', 'he_normal', 'he_uniform', 'glorot_uniform'],
#           'optimizer': [Adam, Nadam, RMSprop],
#           'losses': [binary_crossentropy],
#           'activation': [relu, elu, LeakyReLU],
#           'last_activation': [sigmoid],
#           }

#
# # Mei 22
# # small weight
# # 28329 secs
# params = {'lr': (-6, -1),
#           'dropout': (0, 0.7),
#
#           'batch_size': (100, 2000),
#           'epochs': [100],
#
#           'layer_size': (100, 300),
#           'other_hidden_layers': [0, 1],
#           'shapes': ['funnel'],
#
#           'pos_weight': [2, 5, 10],
#
#           'kernel_initializer': ['normal', 'uniform', 'he_normal', 'he_uniform', 'glorot_uniform'],
#           'optimizer': [Adam, Nadam, RMSprop],
#           'losses': [binary_crossentropy],
#           'activation': [relu, elu, LeakyReLU],
#           'last_activation': [sigmoid],
#           }


# # Mei 23, 24
# # smaller weight
# # 43680 secs
# params = {'lr': (-6, -1),
#           'dropout': (0, 0.7),
#
#           'batch_size': (100, 2000),
#           'epochs': [100],
#
#           'layer_size': (100, 300),
#           'other_hidden_layers': [0, 1],
#           'shapes': ['funnel'],
#
#           'pos_weight': (1, 3),
#
#           'kernel_initializer': ['normal', 'uniform', 'he_normal', 'he_uniform', 'glorot_uniform'],
#           'optimizer': [Adam, Nadam],
#           'losses': [binary_crossentropy],
#           'activation': [elu, LeakyReLU],
#           'last_activation': [sigmoid],
#           }

# Mei_NN_25,  new test data, one hidden layer (benchmark)
# 71950 secs
params = {'lr': (-6, -1),
          'dropout': (0, 0.7),

          'batch_size': (10, 2000),
          'epochs': [100],

          'layer_size': (10, 500),
          'other_hidden_layers': [0],
          'shapes': ['funnel'],

          'pos_weight': [1],

          'kernel_initializer': ['uniform'],
          'optimizer': [Adam, Nadam],
          'losses': [binary_crossentropy],
          'activation': [relu, elu, LeakyReLU],
          'last_activation': [sigmoid],
          }

n_iter = 100

xgb_params = {
    'eta': (-3, -1),
    'max_depth': (3, 10),
    'min_child_weight': (3, 10),
    'subsample': (0, 1),
    'colsample_bytree': (0, 1),
    'eval_metric': ['logloss'],
    'objective': ['binary:logistic'],
    'silent': [1],
}
