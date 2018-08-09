import os
import numpy as np
import pandas as pd
import glob
import warnings
import itertools
import time
import collections

# local items
from scan import Scan
from config import results_dir, temp_dir, data_fp, os_data_fp, data_dir, X_cols
from utils import DataPrep, DataPrepWrapper
from analyze import select_best_model

# sklearn items
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, \
    average_precision_score, f1_score, log_loss
from imblearn.metrics import geometric_mean_score

# keras items
from keras.models import load_model

# xgb
import xgboost as xgb


# Buckets_2
class Group(object):
    def __init__(self):
        pass


class Buckets(object):
    pass


def xgb_predict(model, X, y):
    dtest = xgb.DMatrix(X, label=y)
    prob_pos = model.predict(dtest)
    exp_pos = np.sum(prob_pos)
    return np.sum(y), exp_pos


# dicts
index = {'fico': 6,
         'ltv': 2,
         'oyr': 7,    # with yrs
         'year': 14,  # with yrs
         'dti': 4,
         'age': 12,
         'rel_oupb': 11,
         'purpose': {'C': 25, 'N': 26, 'P': 27},
         'product': {'BP': 44, 'LP': 45},
         'orig_chn': {'B': 21, 'C': 22, 'R': 23},
         'prop_type': {'CO': 30, 'CP': 31, 'MH': 33, 'PUD': 34, 'SF': 35},
         'occ_stat': {'I': 36, 'O': 37, 'S': 39},
         'num_unit': 5,
         'num_bo': 3,
         'fthb': {'Y': 43, 'N': 42},
         'jud_st': {'Y': 41, 'N': 40},
         }


# divide groups
buckets = {'total': ['all'],
           'fico': [(0, 619), (620, 679), (680, 719), (720, 759), (760, 779), (780, np.inf)],
           'ltv': [(0, 80), (80.01, 85), (85.01, 90), (90.01, 95), (95.01, 97), (97, np.inf)],
           'dti': [(0, 14.9), (15, 29.9), (30, 40.9), (41, 42.9), (43, 44.9), (45, np.inf)],
           'age': [(1, 4), (5, 8), (9, 12), (13, 16), (17, 19), (20, 40), (41, np.inf)],
           'rel_oupb': [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 1.3), (1.3, 1.5), (1.5, 1.7), (1.7, 2), (2, np.inf)], # TODO:
           # 'oyr': range(2001, 2017),
           # 'year': range(2001, 2018),
           'purpose': ['C', 'N', 'P'],
           'orig_chn': ['B', 'C', 'R'],
           'prop_type': ['CO/CP', 'MH', 'SF/PUD'],
           'occ_stat': ['I', 'O', 'S'],
           'num_unit': ['multi', 'single'],
           'num_bo': ['multi', 'single'],
           'fthb': ['Y', 'N'],
           'jud_st': ['Y', 'N'],
           }

temp_pdt = {
    'product': ['BP', 'LP'],
}

temp_yrs = {
    'year': range(2001, 2018),
    'oyr': range(2001, 2017),
}

temp_total = {
    'total': ['all'],
}


# 12: 30 mins
# 22: 50 mins
if __name__ == '__main__':
    # select the best model
    round_no = 'xgb'  # 25
    target_metric = 'Loss_val'
    best_k = 2

    model, idx = select_best_model(round_no=round_no, based_on=target_metric, index=None, best_k=best_k)
    print(idx)

    # import xgboost as xgb
    #
    # # re-train the best xgb model
    # df = pd.read_csv(data_fp, low_memory=False)
    # os_df = pd.read_csv(os_data_fp, low_memory=False)
    # is_data = DataPrep(df)
    # X, y = is_data.standardize_Xy()
    # dtrain = xgb.DMatrix(X, label=y)
    #
    # opt_param = {'eta': 0.07707380389113155, 'max_depth': 4, 'min_child_weight': 4, 'subsample': 0.9609851141738814,
    #              'colsample_bytree': 0.48182577339776644, 'eval_metric': 'logloss', 'objective': 'binary:logistic',
    #              'silent': 1}
    # opt_round = 197
    #
    # model = xgb.train(
    #     opt_param,
    #     dtrain,
    #     num_boost_round=opt_round,
    # )

    # fico: 6
    # oltv: 2
    # oyr: 7
    # dti: 4
    # loan_purp_C, N, P, R, U
    # mi_product_BP, LP
    # loan_age_qtr: 13
    # OUPB_Rel:
    # chn
    # prop_type_CO, CP, LH, MH, PU, SF
    # occ_stat_I, O, P, S
    # num_unit: 5
    # num_bo: 3
    # fhb_flag_N, Y
    # judicial_st_N, Y

    # oupb: 1

    start = time.time()
    if not glob.glob(f'{results_dir}\\buckets_{round_no}'):
        os.mkdir(os.path.join(results_dir, f'buckets_{round_no}'))

    for grp_name, grp in temp_total.items():
        # run a loop over vintages
        m_arr = np.zeros((len(grp), 5))
        for v in range(2001, 2017):
            # retrieve X_test, y_test, y_prob, X_test_ns(non-standardized)
            X_test = np.load(f'{temp_dir}\\delyrs\\X_test_delyrs_{v}.pkl')
            y_test = np.load(f'{temp_dir}\\delyrs\\y_test_delyrs_{v}.pkl')
            y_prob = np.load(f'{temp_dir}\\delyrs\\y_prob_delyrs_{v}.pkl')
            # X_test_ns = np.load(f'{temp_dir}\\delyrs\\X_test_ns_delyrs_{v}.pkl')
            X_test_ns = np.load(f'{temp_dir}\\X_test_ns_{v}.pkl')  # to generate bucket result for oyr, year

            # fico bucket
            for i, gp in enumerate(grp):
                if type(gp) is tuple:
                    lb, ub = gp
                    if grp_name is 'rel_oupb':
                        cond = np.logical_and(X_test_ns[:, index[grp_name]] >= lb, X_test_ns[:, index[grp_name]] < ub)
                    elif grp_name is 'ltv' and lb == 97:
                        cond = np.logical_and(X_test_ns[:, index[grp_name]] > lb, X_test_ns[:, index[grp_name]] < ub)
                    else:
                        cond = np.logical_and(X_test_ns[:, index[grp_name]] >= lb, X_test_ns[:, index[grp_name]] <= ub)
                elif type(gp) is int:
                    cond = (X_test_ns[:, index[grp_name]] == gp)
                elif '/' in gp:
                    a, b = gp.split('/')
                    cond = (X_test_ns[:, index[grp_name][a]] + X_test_ns[:, index[grp_name][b]]).astype(bool)
                elif gp == 'multi':
                    cond = X_test_ns[:, index[grp_name]] > 1
                elif gp == 'single':
                    cond = (X_test_ns[:, index[grp_name]] == 1)
                elif gp == 'all':
                    cond = np.ones(X_test_ns.shape[0], dtype=int).astype(bool)
                else:
                    cond = X_test_ns[:, index[grp_name][gp]].astype(bool)

                if np.any(cond):
                    count = np.sum(cond)        # calculate how many loans in this bucket
                    # X_test = np.delete(X_test, [46, 47], 1)
                    d60_act, d60_exp_nn, prob_pos = Scan.model_predict(model[-1], X_test[cond, :], y_test[cond], just_d60=True)

                    # d60_act, d60_exp_nn = xgb_predict(model, X_test[cond, :], y_test[cond])

                    d60_exp_ra = np.sum(y_prob[cond])
                    oupb = np.sum(X_test_ns[cond, 1])
                    m_arr[i, :] += np.array([count, oupb, d60_act, d60_exp_ra, d60_exp_nn])

        df = pd.DataFrame(m_arr, columns=['Count', 'OUPB', 'Act D60', 'Exp D60 radar', 'Exp D60 nn'])
        df['Act D60'] = df['Act D60'] / df['Count']
        df['Exp D60 radar'] = df['Exp D60 radar'] / df['Count']
        df['Exp D60 nn'] = df['Exp D60 nn'] / df['Count']
        df['Abs Diff radar'] = df['Exp D60 radar'] - df['Act D60']
        df['Abs Diff nn'] = df['Exp D60 nn'] - df['Act D60']
        df['Rel Diff radar'] = df['Abs Diff radar'] / df['Act D60']
        df['Rel Diff nn'] = df['Abs Diff nn'] / df['Act D60']

        df = df[['Count', 'OUPB', 'Act D60',
                 'Exp D60 radar', 'Abs Diff radar', 'Rel Diff radar',
                 'Exp D60 nn', 'Abs Diff nn', 'Rel Diff nn']]

        fp = os.path.join(results_dir, f'buckets_{round_no}', f'buckets_{grp_name}.xlsx')
        df.to_excel(fp)
    print(time.time() - start)




# # Buckets_1
# if __name__ == "__main__":
#
#     # get model
#     round_no = 12
#     target_metric = 'Loss_val'
#     best_k = 1
#
#     model, _ = select_best_model(round_no=round_no, based_on=target_metric, index=None, best_k=best_k)
#
#     # collate to excel
#     cols = ['oyr', 'year', 'surv_grp', 'loan_purp', 'fico_grp', 'ltv_grp', 'dti_grp', 'current_cnt', 'current_d60_act', 'current_d60_nn', 'current_d60_radar']
#     df = pd.DataFrame(columns=cols)
#
#     lp_dict = {'C': 27,
#                'N': 28,
#                'P': 29,
#                'R': 30,
#                'U': 31,}
#     # yr = []
#     # for oyr in range(2001, 2017):
#     #     for year in range(oyr, 2018):
#     #         yr.append((oyr, year))
#
#     surv = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 19), (20, 40), (41, np.inf)]
#     loan_purp = ['C', 'N', 'P', 'R', 'U']
#     fico_grp = [(0, 699), (700, 779), (780, np.inf)]
#     ltv_grp = [(0, 80), (80, 85), (85, 90), (90, 95), (95, 97), (97, np.inf)]
#     dti_grp = [(0, 20), (20, 35), (35, 41), (41, 45), (45, np.inf)]
#
#     # buckets = itertools.product(yr, surv, loan_purp, fico_grp, ltv_grp, dti_grp)
#     # the 1st element: (2001, 2001, (1, 4), 'C', (0, 699), (0, 80), (0, 20))
#
#     for v in [2013]:  # range(2001, 2017):
#         start = time.time()
#
#         X_test = np.load(f'{temp_dir}\\X_test_{v}.pkl')
#
#         # get X_test_ns before it is standardized
#         os_data_fp = os.path.join(data_dir, f'Gse_{v}_ltvgt80_v50.csv')
#         is_df = pd.read_csv(data_fp, low_memory=False)
#         os_df = pd.read_csv(os_data_fp, low_memory=False)
#         is_data = DataPrep(is_df)
#         data = DataPrepWrapper(is_data, os_df)
#         X_test_ns = data.os_data.X.values          # TODO: if df is better
#
#         y_test = np.load(f'{temp_dir}\\y_test_{v}.pkl')
#         y_prob = np.load(f'{temp_dir}\\y_prob_{v}.pkl')
#
#         yr = []
#         for year in range(v, 2018):
#             yr.append((v, year))
#
#         buckets = itertools.product(yr, surv, loan_purp, fico_grp, ltv_grp, dti_grp)
#
#         for bucket in buckets:
#             oy_yr, sv, lp, fg, lg, dg = bucket
#             oy, yr = oy_yr
#
#             cond_oy = (X_test_ns[:, 7] == oy)
#             cond_yr = (X_test_ns[:, 14] == yr)
#
#             cond_sv = np.logical_and(X_test_ns[:, 13] >= sv[0], X_test_ns[:, 13] <= sv[1])
#             cond_lp = X_test_ns[:, lp_dict[lp]]
#             cond_fg = np.logical_and(X_test_ns[:, 6] >= fg[0], X_test_ns[:, 6] <= fg[1])
#             cond_lg = np.logical_and(X_test_ns[:, 2] >= lg[0], X_test_ns[:, 2] < lg[1])  # lg[0] <= X_test_ns[:, 2] < lg[1]
#             cond_dg = np.logical_and(X_test_ns[:, 4] >= dg[0], X_test_ns[:, 4] < dg[1])  # dg[0] <= X_test_ns[:, 4] < dg[1]
#
#             cond = np.logical_and(cond_dg, np.logical_and(cond_lg, np.logical_and(cond_fg, np.logical_and(cond_lp, np.logical_and(cond_sv, np.logical_and(cond_oy, cond_yr))))))
#
#             if not np.any(cond):
#                 d60_act, d60_nn, d60_ra = None, None, None
#             else:
#                 print('here')
#                 curr_cnt = np.sum(cond)   # calculate how many loans in this bucket
#                 d60_act, d60_nn = Scan.model_predict(model[0], X_test[cond, :], y_test[cond], just_d60=True)
#                 d60_ra = np.sum(y_prob[cond])
#                 row = [oy, yr] + list(bucket[1:]) + [curr_cnt, d60_act, d60_nn, d60_ra]
#                 df = df.append(dict(zip(cols, row)), ignore_index=True)
#         print(time.time() - start)
#         fp = os.path.join(results_dir, f'buckets_{v}.xlsx')
#         df.to_excel(fp)

