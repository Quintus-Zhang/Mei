import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score, accuracy_score, confusion_matrix, \
    precision_recall_curve, cohen_kappa_score, classification_report
from imblearn.metrics import geometric_mean_score

# TODO: 'RandomForestClassifier' object has no attribute 'decision_function'


def data_prep(df):
    """"""
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
    X = pd.get_dummies(X, columns=CATE_VAR)
    y = df['status'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    return X_train, X_test, y_train, y_test


def model_fit(clf, X_train, X_test, y_train, y_test, threshold=0.5, display=False, save=False, metrics_for_train_set=False):
    """"""
    name_clf = clf.__class__.__name__

    clf.fit(X_train, y_train)

    prob_pos = clf.predict_proba(X_test)[:, 1]
    y_pred = (prob_pos > threshold).astype(int)    # y_pred = clf.predict(X_test)

    # calculate related metrics
    conf_score = clf.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, conf_score)
    fpr, tpr, _ = roc_curve(y_test, conf_score)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_test, conf_score)
    exp_pos = np.sum(prob_pos)
    f_score = f1_score(y_test, y_pred)
    g_mean = geometric_mean_score(y_test, y_pred)

    op = {'Classifier': name_clf,
          'PR_AUC': pr_auc,
          'ROC_AUC': roc_auc,
          'Expeted num of D60-D90': exp_pos,
          'Actual num of D60-D90': np.sum(y_test),
          'F score': f_score,
          'G mean': g_mean}

    # PLOT - confusion matrix
    ax_cm = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")

    # PLOT - ROC_AUC
    fig_ra = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name_clf} - ROC')
    plt.legend(loc="lower right")

    # PLOT - PR_AUC
    fig_pr = plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=1,
             label='PR curve (area = %0.3f)' % pr_auc)
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall(True Positive Rate)')
    plt.ylabel('Precision')
    plt.title(f'{name_clf} - Precision Recall Curve')
    plt.legend(loc="upper right")

    if save:
        # save plots
        ax_cm.get_figure().savefig(f'Results\\{name_clf} - Confusion Matrix')
        fig_ra.savefig(f'Results\\{name_clf} - ROC')
        fig_pr.savefig(f'Results\\{name_clf} - Precision Recall Curve')

    if display:
        fig_ra.show()
        fig_pr.show()

        # Others
        print(f"Accuracy = {accuracy_score(y_test, y_pred)}")
        print("cohen's kappa: ", cohen_kappa_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    if metrics_for_train_set:
        prob_pos_train = clf.predict_proba(X_train)[:, 1]
        y_pred_train = (prob_pos_train > threshold).astype(int)

        conf_score_train = clf.decision_function(X_train)
        precision_train, recall_train, _ = precision_recall_curve(y_train, conf_score_train)
        fpr_train, tpr_train, _ = roc_curve(y_train, conf_score_train)
        roc_auc_train = auc(fpr_train, tpr_train)
        pr_auc_train = average_precision_score(y_train, conf_score_train)
        exp_pos_train = np.sum(prob_pos_train)
        f_score_train = f1_score(y_train, y_pred_train)
        g_mean_train = geometric_mean_score(y_train, y_pred_train)

        op_train = {'Classifier': name_clf + '_train',
                    'PR_AUC': pr_auc_train,
                    'ROC_AUC': roc_auc_train,
                    'Expeted num of D60-D90': exp_pos_train,
                    'Actual num of D60-D90': np.sum(y_train),
                    'F score': f_score_train,
                    'G mean': g_mean_train}

    return op, op_train


