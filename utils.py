import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.callbacks import Callback
from keras.layers import Dense, Dropout


def data_prep(raw_df):
    """"""
    df = raw_df.copy()

    JUD_ST = ('CT', 'DE', 'FL', 'IL', 'IN', 'KS', 'KY', 'LA', 'ME', 'MA',
              'NE', 'NJ', 'NM', 'NY', 'ND', 'OH', 'OK', 'PA', 'SC', 'SD',
              'VT', 'WI')
    CATE_VAR = ['orig_chn', 'loan_purp', 'prop_type', 'occ_stat', 'judicial_st', 'fhb_flag']

    # drop columns
    df.drop(['loan_id', 'status_prev', 'msa'], axis=1, inplace=True)

    # drop all the observation with missing value
    df.dropna(how='any', inplace=True)

    # create a new feature based on prop_state
    df.loc[:, 'judicial_st'] = df['prop_state'].apply(lambda x: 'Y' if x in JUD_ST else 'N')
    df.drop(['prop_state'], axis=1, inplace=True)

    # convert status to 0 or 1
    df.loc[:, 'status'] = df['status'].apply(lambda x: int(x == 'D60-D90'))

    X = df.drop(['status'], axis=1).copy()
    num_feats = list(set(X.columns) - set(CATE_VAR))
    X = pd.get_dummies(X, columns=CATE_VAR)
    y = df['status'].copy()

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=1)

    # mean is computed only over the training data and then subtracted equally from all splits(train/val/test)
    scaler = StandardScaler().fit(X_train.loc[:, num_feats])
    pd.set_option('mode.chained_assignment', None)    # TODO
    X_train.loc[:, num_feats] = scaler.transform(X_train.loc[:, num_feats])
    X_val.loc[:, num_feats] = scaler.transform(X_val.loc[:, num_feats])
    X_test.loc[:, num_feats] = scaler.transform(X_test.loc[:, num_feats])
    return X_train.values, X_val.values, X_test.values, y_train.values, y_val.values, y_test.values


class ModelCheckpointRtnBest(Callback):
    """Get the best model at the end of training.
	# Arguments
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        mode: one of {auto, min, max}.
            The decision
            to overwrite the current stored weights is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
	# Example
		callbacks = [GetBest(monitor='val_acc', verbose=1, mode='max')]
		mode.fit(X, y, validation_data=(X_eval, Y_eval),
                 callbacks=callbacks)
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 mode='auto', period=1):
        super(ModelCheckpointRtnBest, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.period = period
        self.best_epochs = 0
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('GetBest mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can pick best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' storing weights.'
                              % (epoch + 1, self.monitor, self.best,
                                 current))
                    self.best = current
                    self.best_epochs = epoch + 1
                    self.best_weights = self.model.get_weights()
                    # self.model.save(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve' %
                              (epoch + 1, self.monitor))

    def on_train_end(self, logs=None):
        if self.verbose > 0:
            print('Using epoch %05d with %s: %0.5f' % (self.best_epochs, self.monitor,
                                                       self.best))
        self.model.set_weights(self.best_weights)




