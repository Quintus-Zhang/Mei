import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.callbacks import Callback
from keras.layers import Dense, Dropout


class DataPrep(object):
    CATE_VAR = ['orig_chn', 'loan_purp', 'prop_type', 'occ_stat', 'judicial_st', 'fhb_flag']
    JUD_ST = ('CT', 'DE', 'FL', 'IL', 'IN', 'KS', 'KY', 'LA', 'ME', 'MA',
              'NE', 'NJ', 'NM', 'NY', 'ND', 'OH', 'OK', 'PA', 'SC', 'SD',
              'VT', 'WI')

    def __init__(self, df):
        self.df = df
        self._X, self._y = self.clean_data()

        self._num_feats = self._num_feats()

        self.X = self._one_hot_coder()
        self.y = self._y

    def clean_data(self):
        df = self.df.copy()

        # drop columns
        try:
            df.drop(['loan_id', 'status_prev', 'msa'], axis=1, inplace=True)
        except:
            df.drop(['status_prev', 'msa'], axis=1, inplace=True)

        # drop all the observation with missing value
        df.dropna(how='any', inplace=True)

        # create a new feature based on prop_state
        df.loc[:, 'judicial_st'] = df['prop_state'].apply(lambda x: 'Y' if x in self.JUD_ST else 'N')
        df.drop(['prop_state'], axis=1, inplace=True)

        # convert status to 0 or 1
        df.loc[:, 'status'] = df['status'].apply(lambda x: int(x == 'D60-D90'))

        X = df.drop(['status'], axis=1).copy()
        y = df['status'].copy()
        return X, y

    def _num_feats(self):
        return list(set(self._X.columns) - set(self.CATE_VAR))

    def _one_hot_coder(self):
        return pd.get_dummies(self._X, columns=self.CATE_VAR)

    def split_and_standardize(self):
        return self.standardize(*self.split(method='train_val_test_split'))

    def split(self, method='train_val_test_split'):
        if method is 'train_val_test_split':
            X_train_val, X_test, y_train_val, y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=111)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.33, random_state=22)
            return X_train, X_val, X_test, y_train, y_val, y_test
        elif method is 'train_val_split':
            X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.3, random_state=0)
            return X_train, X_val, y_train, y_val
        else:
            warnings.warn('Not implemented')

    def standardize(self, X_train, X_val, X_test, y_train, y_val, y_test):
        # mean is computed only over the training data and then subtracted equally from all splits(train/val/test)
        scaler = StandardScaler().fit(X_train.loc[:, self._num_feats])
        pd.set_option('mode.chained_assignment', None)                                          # TODO: ugly here
        X_train.loc[:, self._num_feats] = scaler.transform(X_train.loc[:, self._num_feats])
        X_val.loc[:, self._num_feats] = scaler.transform(X_val.loc[:, self._num_feats])
        X_test.loc[:, self._num_feats] = scaler.transform(X_test.loc[:, self._num_feats])
        return X_train.values, X_val.values, X_test.values, y_train.values, y_val.values, y_test.values
        # TODO: test dataset has different shape with train and val


class DataPrepWrapper(object):
    COLS = ['loan_id', 'orig_chn', 'orig_rt', 'orig_upb', 'oltv', 'num_bo', 'dti', 'loan_purp', 'prop_type', 'num_unit',
            'occ_stat', 'prop_state', 'msa', 'fico', 'oyr', 'oqtr', 'ind_ede', 'pmms_o', 'avg_upb', 'OUPB_Rel',
            'fhb_flag', 'loan_age_qtr', 'year', 'qtr', 'status', 'status_prev', 'PMMS', 'HPI_O', 'HPI', 'lag_ur',
            'CUPB_calc', 'Orig_value', 'CLTV']
    rename = {'oupb_rel': 'OUPB_Rel', 'pmms': 'PMMS', 'hpi_o': 'HPI_O',
              'hpi': 'HPI', 'lag_ur': 'ur', 'cupb_calc': 'CUPB_calc',
              'orig_value': 'Orig_value', 'cltv': 'CLTV'}

    def __init__(self, is_data, os_df):
        self.is_data = is_data
        self.os_df = os_df

        self.preproced_os_df, self.pos_prob = self._preproc_os_data()
        self.os_data = DataPrep(self.preproced_os_df)
        self._homo_ios_data()

    def _preproc_os_data(self):
        self.os_df.rename(str.lower, axis='columns', inplace=True)
        cols = [col.lower() for col in self.COLS]
        df = self.os_df[cols].copy()
        df.rename(self.rename, axis='columns', inplace=True)

        pos_prob = self.os_df['p_current_d60'].copy()
        return df, pos_prob

    def _homo_ios_data(self):
        # homo the columns
        print(list(set(self.is_data.X.columns) - set(self.os_data.X.columns)))
        for col in list(set(self.is_data.X.columns) - set(self.os_data.X.columns)):
            self.os_data.X[col] = 0
            self.os_data.X[col] = self.os_data.X[col].astype('uint8')  # TODO: check data type
        # order the columns
        self.os_data.X = self.os_data.X[self.is_data.X.columns]

    def split_and_standardize(self):
        X_train, X_val, y_train, y_val = self.is_data.split(method='train_val_split')
        return self.is_data.standardize(X_train, X_val, self.os_data.X, y_train, y_val, self.os_data.y)


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


def oos_data_prep(raw_df):
    COLS = ['orig_chn', 'orig_rt', 'orig_upb', 'oltv', 'num_bo', 'dti', 'loan_purp', 'prop_type', 'num_unit',
            'occ_stat', 'prop_state', 'msa', 'fico', 'oyr', 'oqtr', 'ind_ede', 'pmms_o', 'avg_upb', 'OUPB_Rel',
            'fhb_flag', 'loan_age_qtr', 'year', 'qtr', 'status', 'status_prev', 'PMMS', 'HPI_O', 'HPI', 'lag_ur',
            'CUPB_calc', 'Orig_value', 'CLTV']
    raw_df.rename(str.lower, axis='columns', inplace=True)
    cols = [col.lower() for col in COLS]
    df = raw_df[cols].copy()

    pos_prob = raw_df['p_current_d60'].copy()

    return df, pos_prob


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
                    self.best_val_loss = logs['val_loss']
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
        filepath = self.filepath.format(epoch=self.best_epochs, val_loss=self.best_val_loss)
        self.model.save(filepath, overwrite=True)    # save the best model over the training path of a certain combo


# expected_positives_loss
def expected_positives_loss(y_true, y_pred):
    return abs(np.sum(y_true) - np.sum(y_pred))
