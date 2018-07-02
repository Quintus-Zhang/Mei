import numpy as np
import multiprocessing
import csv
import glob
import os
import re

# sklearn items
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, f1_score

#
from imblearn.metrics import geometric_mean_score

# TODO: clear_session
# TODO: separate _output_setup method as a class
# TODO: clever way to create results header


class Scan(object):
    def __init__(self, X_train, y_train, X_val, y_val, params_grid, dataset_name, model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.params_grid = params_grid
        self.dataset_name = dataset_name
        self.model = model
        self.result_dir = '.\\Results'
        self.round_dir = ''
        self.round_fp = ''

        # to set the round_dir and the round_fp, to make a round directory under Results directory
        self._output_setup()

        #
        self._write_results_header()

        # self.params_grid = self.get_params_grid()
        self._run_search()
        # self.mp_handler()

    def _run_search(self):
        with open(self.round_fp, 'a', newline='') as f:
            res_writer = csv.writer(f, dialect='excel', delimiter=',')
            for idx, params in enumerate(self.params_grid):
                # tf.reset_default_graph()
                result = self.worker((idx, params))
                print(f'Saving results to {self.round_fp}')
                res_writer.writerow(result)
                f.flush()

    def worker(self, args):
        idx, params = args
        trained_model = self.model(self.X_train, self.y_train, self.X_val, self.y_val, params, idx, self.round_dir)
        metrics_tra = self._model_predict(trained_model, self.X_train, self.y_train)
        metrics_val = self._model_predict(trained_model, self.X_val, self.y_val)
        return [str(idx)] + self._collect_results(metrics_tra) + self._collect_results(metrics_val) + self._collect_results(params)

    def mp_handler(self):
        cores = multiprocessing.cpu_count()
        with multiprocessing.Pool(cores) as p:
            with open(self.round_fp, 'a', newline='') as f:
                res_writer = csv.writer(f, dialect='excel', delimiter=',')
                for result in p.imap(self.worker, enumerate(self.params_grid)):      # a queue defined by pool object
                    print(f'Saving results to {self.round_fp}')
                    res_writer.writerow(result)
                    f.flush()

    @staticmethod
    def _model_predict(model, X, y):
        # evaluate
        loss = model.evaluate(X, y, batch_size=X.shape[0])

        # predict
        prob_pos = model.predict(X, batch_size=X.shape[0], verbose=0)    # TODO: batch_size
        vfunc = np.vectorize(lambda x: 1 if x > 0.05 else 0)
        y_pred = vfunc(prob_pos).ravel()

        # calculate performance metrics
        precision, recall, _ = precision_recall_curve(y, prob_pos)
        fpr, tpr, _ = roc_curve(y, prob_pos)
        roc_auc = auc(fpr, tpr)
        pr_auc = average_precision_score(y, prob_pos)
        exp_pos = np.sum(prob_pos)
        f_score = f1_score(y, y_pred)
        g_mean = geometric_mean_score(y, y_pred)

        metrics = {'loss': loss,
                   'PR_AUC': pr_auc,
                   'ROC_AUC': roc_auc,
                   'F_score': f_score,
                   'G_mean': g_mean,
                   'Expeted_#_D60': exp_pos,
                   'Actual_#_D60': np.sum(y), }
        return metrics

    def _write_results_header(self):
        header = ['Index',
                  'Loss_train', 'PR_AUC_train', 'ROC_AUC_train', 'F_score_train', 'G_mean_train', 'Expeted_#_D60_train',
                  'Actual_#_D60_train',
                  'Loss_val', 'PR_AUC_val', 'ROC_AUC_val', 'F_score_val', 'G_mean_val', 'Expeted_#_D60_val',
                  'Actual_#_D60_val',
                  'lr', 'dropout', 'other_hidden_layers', 'layer_size', 'batch_size', 'epochs', 'shapes',
                  'kernel_initializer', 'optimizer', 'losses', 'activation', 'last_activation']
        with open(self.round_fp, 'w', newline='') as f:
            res_writer = csv.writer(f, dialect='excel', delimiter=',')
            res_writer.writerow(header)

    @staticmethod
    def _collect_results(results):
        op = []
        for key in list(results.keys()):
            op.append(results[key])
        return op  # ",".join(str(i) for i in op)

    def _output_setup(self):
        rounds = glob.glob(f'{self.result_dir}\\{self.dataset_name}_*')
        if not rounds:
            experiment_no = 1
        else:
            sorted_rounds = sorted(rounds, key=lambda x: int(re.search(r'\d+$', x).group()))  # sort rounds by index
            experiment_no = int(sorted_rounds[-1].split('_')[-1]) + 1
        self.round_dir = f'{self.result_dir}\\{self.dataset_name}_{experiment_no}'
        os.mkdir(self.round_dir)
        self.round_fp = f'{self.round_dir}\\{self.dataset_name}_{experiment_no}.csv'


    # # disposable
    # def get_params_grid(self):
    #     params_name = list(self.params.keys())
    #     combos = itertools.product(*(self.params[key] for key in params_name))   # iterable of tuple, each tuple is one combination
    #     params_grid = [dict(zip(params_name, combo)) for combo in combos]        # list of dict
    #     return params_grid