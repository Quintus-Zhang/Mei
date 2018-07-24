import os
import numpy as np
import pandas as pd
import glob
import warnings

# keras items
from keras.models import load_model

# local items
from scan import Scan
from config import results_dir, temp_dir, data_fp, os_data_fp
from utils import DataPrep, DataPrepWrapper

# sklearn items
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, \
    average_precision_score, f1_score, log_loss
from imblearn.metrics import geometric_mean_score

# read the 'Mei_NN_*.csv' file
# load weights of the best model based on a specific metric
# - return the model with lowest loss_val
# - return the model with lowest Diff_#_D60_val
# use which test dataset
# - that from current 1m dataset
# - that from extra source

# class Analyze(object):
#     def __init__(self, vintage, round_no, target_metric, best_k, index):
#         self.vintage = vintage
#         self.round_no = round_no
#         self.target_metric = target_metric
#         self.best_k = best_k
#         self.index = index
#
#     def set_X_test(self):
#         # retrieve test dataset
#         self.X_test = np.load(f'{temp_dir}\\X_test_{yr}.pkl')
#
#     def set_y_test(self):
#         self.y_test = np.load(f'{temp_dir}\\y_test_{yr}.pkl')
#
#     def set_y_prob(self):
#         self.y_prob = np.load(f'{temp_dir}\\y_prob_{yr}.pkl')


def select_best_model(round_no=None, based_on='Loss_val', index=None, best_k=1):
    """ Select best model based on the given metric

    :param round_no:
    :param based_on:
    :return:
    """

    if round_no is None:
        warnings.warn('round_no has to be specified')
        return

    round_dir = os.path.join(results_dir, f'Mei_NN_{round_no}')
    round_fp = os.path.join(round_dir, f'Mei_NN_{round_no}.csv')

    if based_on is 'index':
        return load_model(glob.glob(f'{round_dir}\\{index}_*')[0])

    # read the 'Mei_NN_*.csv' file
    res = pd.read_csv(round_fp)

    # search for file name of the best model based on a specific metric
    if based_on in ['Loss_val', 'Diff_D60_val']:
        params_idx = res.loc[:, based_on].sort_values().index[:best_k]
    elif based_on in ['PR_AUC_val', 'ROC_AUC_val', 'F_score_val', 'G_mean_val']:
        params_idx = res.loc[:, based_on].sort_values(ascending=False).index[:best_k]
    else:
        warnings.warn(f'Cannot select the best model based on {based_on}')

    models = []
    for idx in params_idx:
        best_model_fp = glob.glob(f'{round_dir}\\{idx}_*')[0]
        best_model = load_model(best_model_fp)
        models.append(best_model)
    return models, params_idx


def calc_metrics_radar(y_true, y_prob):
    # calculate the metrics for prediction probabilities from RaDaR
    vfunc = np.vectorize(lambda x: 1 if x > 0.05 else 0)
    y_pred = vfunc(y_prob).ravel()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_true, y_prob)
    exp_pos = np.sum(y_prob)
    f_score = f1_score(y_true, y_pred)
    g_mean = geometric_mean_score(y_true, y_pred)

    loss = log_loss(y_true, y_prob)

    metrics = {'Loss': loss,
               'PR_AUC': pr_auc,
               'ROC_AUC': roc_auc,
               'F_score': f_score,
               'G_mean': g_mean,
               'Expeted_#_D60': exp_pos,
               'Actual_#_D60': np.sum(y_true),
               'Diff_D60': abs(np.sum(y_true) - exp_pos),
               'Ratio_D60': np.sum(y_true) / exp_pos,
               }
    return metrics


if __name__ == "__main__":
    # get model
    round_no = 14
    target_metric = 'Loss_val'
    best_k = 5

    models, idx = select_best_model(round_no=round_no, based_on=target_metric, index=None, best_k=best_k)

    # collate to excel
    cols = ['Vintage', 'Model', 'Index', 'Loss', 'PR_AUC', 'ROC_AUC', 'F_score', 'G_mean',
            'Expeted_#_D60', 'Actual_#_D60', 'Diff_D60', 'Ratio_D60']
    cmp_df = pd.DataFrame(columns=cols)

    # predict
    for v in range(2001, 2017):
        print(v)
        X_test = np.load(f'{temp_dir}\\X_test_{v}.pkl')
        y_test = np.load(f'{temp_dir}\\y_test_{v}.pkl')
        y_prob = np.load(f'{temp_dir}\\y_prob_{v}.pkl')

        # radar metrics
        metrics_ra = calc_metrics_radar(y_test, y_prob)
        metrics_ra.update({'Vintage': v, 'Model': 'RaDaR', 'Index': None})
        cmp_df = cmp_df.append(metrics_ra, ignore_index=True)

        # for each model, calculate the metrics
        # prob = np.zeros_like(y_prob)
        # for model, id in zip(models, idx):
        #     prob += np.squeeze(model.predict(X_test, batch_size=X_test.shape[0], verbose=0))
        #     # metrics_nn = Scan.model_predict(model, X_test, y_test)
        #     # metrics_nn.update({'Vintage': v, 'Model': f'Mei_NN_{round_no}', 'Index': id})
        #     # cmp_df = cmp_df.append(metrics_nn, ignore_index=True)
        #
        # prob /= best_k
        # vfunc = np.vectorize(lambda x: 1 if x > 0.05 else 0)
        # y_pred = vfunc(prob).ravel()
        #
        # # calculate performance metrics
        # precision, recall, _ = precision_recall_curve(y_test, prob)
        # fpr, tpr, _ = roc_curve(y_test, prob)
        # roc_auc = auc(fpr, tpr)
        # pr_auc = average_precision_score(y_test, prob)
        # exp_pos = np.sum(prob)
        # f_score = f1_score(y_test, y_pred)
        # g_mean = geometric_mean_score(y_test, y_pred)
        #
        # metrics_nn = {'Loss': None,
        #            'PR_AUC': pr_auc,
        #            'ROC_AUC': roc_auc,
        #            'F_score': f_score,
        #            'G_mean': g_mean,
        #            'Expeted_#_D60': exp_pos,
        #            'Actual_#_D60': np.sum(y_test),
        #            'Diff_D60': abs(np.sum(y_test) - exp_pos),
        #            'Ratio_D60': np.sum(y_test) / exp_pos,
        #            }
        # metrics_nn.update({'Vintage': v, 'Model': f'Mei_NN_{round_no}', 'Index': None})
        # cmp_df = cmp_df.append(metrics_nn, ignore_index=True)

        # for model, id in zip(models, idx):
        #     metrics_nn = Scan.model_predict(model, X_test, y_test)
        #     metrics_nn.update({'Vintage': v, 'Model': f'Mei_NN_{round_no}', 'Index': id})
        #     cmp_df = cmp_df.append(metrics_nn, ignore_index=True)

    cmp_fp = os.path.join(results_dir, f'cmp_round{round_no}_{target_metric}_best{best_k}_loss.xlsx')
    cmp_df.to_excel(cmp_fp)


# round_3
# not good, hold-out validation is not enough

