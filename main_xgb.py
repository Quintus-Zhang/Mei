import time
import pandas as pd
import numpy as np

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

# local items
from config import *
from utils import DataPrep, DataPrepWrapper
from params import ParamsGridSearch, ParamsRandomSearch

# TODO: params, utils
if __name__ == "__main__":
    start = time.time()
    # 200 rounds, 50 iters, 31749 secs
    # val_loss: 0.0133482
    # {'eta': 0.08395946869088414, 'max_depth': 3, 'min_child_weight': 9, 'subsample': 0.9662365667946802,
    # 'colsample_bytree': 0.3750271939029731, 'eval_metric': 'logloss', 'objective': 'binary:logistic', 'silent': 1}

    # 79395
    # 0.0133464
    # 197
    # {'eta': 0.07707380389113155, 'max_depth': 4, 'min_child_weight': 4, 'subsample': 0.9609851141738814, 'colsample_bytree': 0.48182577339776644, 'eval_metric': 'logloss', 'objective': 'binary:logistic', 'silent': 1}

    df = pd.read_csv(data_fp, low_memory=False)
    os_df = pd.read_csv(os_data_fp, low_memory=False)
    is_data = DataPrep(df)
    X, y = is_data.standardize_Xy()

    dtrain = xgb.DMatrix(X, label=y)
    prs = ParamsRandomSearch(xgb_params, n_iter=n_iter)

    num_boost_round = 200
    opt_loss = np.inf
    checkpoint = list(range(25, n_iter, 25))
    for i, param in enumerate(prs.params_grid):
        print(i)
        cv_results = xgb.cv(
            param,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['logloss'],
            early_stopping_rounds=10
        )

        best_loss = cv_results['test-logloss-mean'].min()
        best_round = cv_results['test-logloss-mean'].argmin()

        if best_loss < opt_loss:
            opt_loss = best_loss
            opt_round = best_round
            opt_param = param

        if i in checkpoint:
            print(opt_loss)  # cv loss
            print(opt_round)
            print(opt_param)

            best_model = xgb.train(
                opt_param,
                dtrain,
                num_boost_round=opt_round,
            )

            fp = os.path.join(results_dir, 'xgb', f'best_xgb_cp{i}.model')
            best_model.save_model(fp)

    print(time.time() - start)
