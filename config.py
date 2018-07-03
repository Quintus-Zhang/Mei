# Keras items
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import Adam, Nadam
from keras.activations import relu, elu, sigmoid
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping
from keras import backend as K

# tensorflow items
import tensorflow as tf

# local items
from utils import ModelCheckpointRtnBest
from layers import add_other_hidden_layers


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
    # TODO: try to return the index of epoch where model training is stopped
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


params = {'lr': (-6, 1),                    # log scale for lr
          'dropout': (0, 0.5),

          'batch_size': (100, 2000),
          'epochs': [100],

          'layer_size': (10, 200),
          'other_hidden_layers': [0, 1],
          'shapes': ['funnel'],

          'kernel_initializer': ['normal'],
          'optimizer': [Adam],
          'losses': [binary_crossentropy],
          'activation': [relu],
          'last_activation': [sigmoid]
          }

n_iter = 2
