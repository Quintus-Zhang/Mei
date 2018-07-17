import os
# Keras items
from keras.models import Sequential
from keras.layers import Dropout, Dense, LeakyReLU
from keras.optimizers import Adam, Nadam, RMSprop
from keras.activations import relu, elu, sigmoid
from keras.losses import binary_crossentropy, logcosh
from keras.callbacks import EarlyStopping

# tensorflow items
from keras import backend as K
import tensorflow as tf
import tensorflow as tf

# local items
from utils import ModelCheckpointRtnBest, expected_positives_loss
from layers import add_other_hidden_layers

# set up paths of directories and files
base_dir = os.path.dirname(__file__)
results_dir = os.path.join(base_dir, 'Results')
temp_dir = os.path.join(base_dir, 'Temp')
data_dir = os.path.join(base_dir, 'Data')
data_fp = os.path.join(data_dir, 'Gse_panel_current_sample_raw.csv')
os_data_fp = os.path.join(data_dir, 'Gse_2016_ltvgt80_v50.csv')


#########################################
#          Set up your model            #
#########################################
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
                  optimizer=params['optimizer']())

    # set up callbacks
    cp_fp = f'{cp_dir}\\{params_idx}' + '_best_model_{epoch:02d}_{val_loss:.5f}.hdf5'
    check_pointer = ModelCheckpointRtnBest(filepath=cp_fp, monitor='val_loss', mode='min', verbose=0)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=0, mode='min')
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

params = {'lr': (-6, -1),                    # log scale for lr
          'dropout': (0, 0.5),

          'batch_size': (10, 2000),
          'epochs': [100],

          'layer_size': (10, 500),
          'other_hidden_layers': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
          'shapes': ['funnel'],

          'pos_weight': [1],

          'kernel_initializer': ['uniform'],
          'optimizer': [Adam, Nadam, RMSprop],
          'losses': [binary_crossentropy],
          'activation': [relu, elu, LeakyReLU],
          'last_activation': [sigmoid],
          }

n_iter = 100


