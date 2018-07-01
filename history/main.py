import os
import time
import pickle as pk
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from history.utility import data_prep, model_fit

if __name__ == '__main__':
    base_path = os.path.dirname(__file__)
    results_path = os.path.join(base_path, 'Results')
    data_fp = os.path.join(base_path, 'Data', 'Gse_panel_current_sample_raw.csv')
    metrics_fp = os.path.join(results_path, 'metrics_results.xlsx')

    df = pd.read_csv(data_fp)
    X_train, X_test, y_train, y_test = data_prep(df)

    clf_list = [
        LogisticRegression(),
        # AdaBoostClassifier(DecisionTreeClassifier(random_state=11, max_features="auto")),
        AdaBoostClassifier(LogisticRegression()),
        GradientBoostingClassifier(),
    ]

    params_grid = {
        'LogisticRegression': dict(C=[0.001, 0.01, 0.1, 1, 10, 100, 1000]),
        # 'AdaBoostClassifier': dict(base_estimator__criterion=["gini", "entropy"],
        #                            base_estimator__splitter=["best", "random"],
        #                            base_estimator__max_depth=list(range(2, 16, 2)),
        #                            base_estimator__min_samples_split=list(range(600, 1601, 200)),
        #                            base_estimator__min_samples_leaf=list(range(30, 71, 10)),
        #                            n_estimators=list(range(20, 201, 20))),
        'GradientBoostingClassifier': dict(n_estimators=list(range(20, 201, 20)),
                                           max_depth=list(range(2, 16, 3)),
                                           min_samples_split=list(range(600, 1601, 200))),
        'AdaBoostClassifier': dict(base_estimator__C=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                   n_estimators=list(range(40, 301, 40)),
                                   learning_rate=[0.1]),
    }

    n_iter_dict = {
        'LogisticRegression': 7,
        'GradientBoostingClassifier': 20,
        'AdaBoostClassifier': 10
    }

    metrics_df = pd.DataFrame(columns=['Classifier', 'PR_AUC', 'ROC_AUC',
                                       'Expeted num of D60-D90', 'F score', 'G mean'])

    best_params_list = []
    for clf in clf_list:
        start = time.time()
        cv_gen = StratifiedKFold(n_splits=5, random_state=0)
        hp = RandomizedSearchCV(clf, params_grid[clf.__class__.__name__], n_iter=n_iter_dict[clf.__class__.__name__],
                                refit=True, scoring='f1', cv=cv_gen, n_jobs=4)
        hp.fit(X_train, y_train)
        clf.set_params(**hp.best_params_)

        metrics_dict, metrics_dict_train = model_fit(clf, X_train, X_test, y_train, y_test,
                                                     threshold=0.05, display=False, save=True,
                                                     metrics_for_train_set=True)
        print(metrics_dict)
        print(hp.best_params_)
        metrics_df = metrics_df.append(metrics_dict, ignore_index=True)
        metrics_df = metrics_df.append(metrics_dict_train, ignore_index=True)
        best_params_list.append(hp.best_params_)
        print(f'{clf.__class__.__name__}: {time.time()-start:.2f} seconds')

    # dump results
    metrics_df.set_index('Classifier', inplace=True)
    metrics_df.to_excel(metrics_fp)
    with open('best params.pkl', 'wb') as f:
        pk.dump(best_params_list, f)
