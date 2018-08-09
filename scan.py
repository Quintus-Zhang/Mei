import numpy as np
import multiprocessing as mp
import csv
import glob
import os
import re

# sklearn items
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score

#
from imblearn.metrics import geometric_mean_score
from keras import backend as K
import tensorflow as tf

# TODO: clear_session or del model
# TODO: separate _output_setup method as a class
# TODO: clever way to create results header


class Scan(object):
    def __init__(self, X_train, y_train, X_val, y_val, params_search, dataset_name, model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.params_grid = params_search.params_grid
        self.params_name = params_search.params_name
        self.dataset_name = dataset_name
        self.model = model
        self.result_dir = '.\\Results'
        self.round_dir = ''
        self.round_fp = ''

        # to set the round_dir and the round_fp, to make a round directory under Results directory
        self._output_setup()

        #
        self._write_results_header()

        # self._run_search()
        self.mp_handler()

    def _run_search(self):
        """
            For-loop to iterate over different combs of parameters
        """
        with open(self.round_fp, 'a', newline='') as f:
            res_writer = csv.writer(f, dialect='excel', delimiter=',')
            for idx, params in enumerate(self.params_grid):
                # tf.reset_default_graph()
                result = self.worker((idx, params))
                # K.clear_session()
                print(f'Saving results to {self.round_fp}')
                res_writer.writerow(result)
                f.flush()

    def worker(self, args):
        """
            Train the model and predict on the validation set
        :return: a list, containing the metrics for the training set and validation set, and the corresponding combo of
        parameter
        """
        idx, params = args
        trained_model, stopped_epoch = self.model(self.X_train, self.y_train,
                                                  self.X_val, self.y_val, params, idx, self.round_dir)
        metrics_tra = self.model_predict(trained_model, self.X_train, self.y_train)
        metrics_val = self.model_predict(trained_model, self.X_val, self.y_val)
        return [str(idx)] \
            + self._collect_results(metrics_tra) \
            + self._collect_results(metrics_val) \
            + [str(stopped_epoch)] \
            + self._collect_results(params)

    def mp_handler(self):
        """
            Multiprocess version to iterate over different combs of parameters
        """
        cores = mp.cpu_count()
        with mp.Pool(cores) as p:
            with open(self.round_fp, 'a', newline='') as f:
                res_writer = csv.writer(f, dialect='excel', delimiter=',')
                for result in p.imap(self.worker, enumerate(self.params_grid)):
                    print(f'Saving results to {self.round_fp}')
                    res_writer.writerow(result)
                    f.flush()

    @staticmethod
    def model_predict(model, X, y, just_d60=False):
        """
            perform the model prediction and calculate various metrics

        :param model: trained model
        :param X: a numpy array, data feed to the model
        :param y: a numpy array, ground true label
        :param just_d60: flag
        :return: a dict
        """
        if just_d60:
            prob_pos = model.predict(X, batch_size=X.shape[0], verbose=0)
            exp_pos = np.sum(prob_pos)
            return np.sum(y), exp_pos, prob_pos

        # evaluate
        loss = model.evaluate(X, y, batch_size=X.shape[0], verbose=0)

        # predict
        prob_pos = model.predict(X, batch_size=X.shape[0], verbose=0)    # TODO: batch_size
        vfunc = np.vectorize(lambda x: 1 if x > 0.05 else 0)
        y_pred = vfunc(prob_pos).ravel()

        # calculate performance metrics
        # precision, recall, _ = precision_recall_curve(y, prob_pos)
        fpr, tpr, _ = roc_curve(y, prob_pos)
        roc_auc = auc(fpr, tpr)
        pr_auc = average_precision_score(y, prob_pos)
        exp_pos = np.sum(prob_pos)
        f_score = f1_score(y, y_pred)
        g_mean = geometric_mean_score(y, y_pred)

        metrics = {'Loss': loss,
                   'PR_AUC': pr_auc,
                   'ROC_AUC': roc_auc,
                   'F_score': f_score,
                   'G_mean': g_mean,
                   'Expeted_#_D60': exp_pos,
                   'Actual_#_D60': np.sum(y),
                   'Diff_D60': abs(np.sum(y) - exp_pos),
                   'Ratio_D60': np.sum(y) / exp_pos,
                   }
        return metrics

    def _write_results_header(self):
        """
            Write the header of the output file
        """
        header = ['Index',
                  'Loss_train', 'PR_AUC_train', 'ROC_AUC_train', 'F_score_train', 'G_mean_train', 'Expeted_#_D60_train',
                  'Actual_#_D60_train', 'Diff_D60_train', 'Ratio_D60_train',
                  'Loss_val', 'PR_AUC_val', 'ROC_AUC_val', 'F_score_val', 'G_mean_val', 'Expeted_#_D60_val',
                  'Actual_#_D60_val', 'Diff_D60_val', 'Ratio_D60_val',
                  'Stopped_Epochs'] + self.params_name
        with open(self.round_fp, 'w', newline='') as f:
            res_writer = csv.writer(f, dialect='excel', delimiter=',')
            res_writer.writerow(header)

    @staticmethod
    def _collect_results(results):
        op = []
        for key in list(results.keys()):
            op.append(results[key])
        return op

    def _output_setup(self):
        rounds = glob.glob(f'{self.result_dir}\\{self.dataset_name}_*')
        if not rounds:
            round_no = 1
        else:
            sorted_rounds = sorted(rounds, key=lambda x: int(re.search(r'\d+$', x).group()))  # sort rounds by index
            round_no = int(sorted_rounds[-1].split('_')[-1]) + 1
        self.round_dir = f'{self.result_dir}\\{self.dataset_name}_{round_no}'
        os.mkdir(self.round_dir)
        self.round_fp = f'{self.round_dir}\\{self.dataset_name}_{round_no}.csv'



