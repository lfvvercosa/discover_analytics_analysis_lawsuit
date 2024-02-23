import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def run_naive(df, n_splits):

    n_splits = 10
    gt = 'TEMPO_PROCESSUAL_TOTAL_DIAS'

    feat = [c for c in df.columns if c != gt]

    X = df[feat].to_numpy()
    y = df[gt].to_numpy()

    skf = KFold(n_splits=n_splits, shuffle=True, random_state=3)
    skf.get_n_splits(X, y)

    mse_naive = []
    r2_naive = []
    mae_naive = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_naive = np.full(len(y_test),y_train.mean())

        mse_naive.append(mean_squared_error(y_test, y_naive))
        r2_naive.append(r2_score(y_test, y_naive))
        mae_naive.append(mean_absolute_error(y_test, y_naive))

    naive_mse_mean = sum(mse_naive) / len(mse_naive)
    naive_r2_mean = sum(r2_naive) / len(r2_naive)
    naive_mae_mean = sum(mae_naive) / len(mae_naive)

    naive_mse_var = sum([(x - naive_mse_mean) ** 2 for x in mse_naive]) / \
        len(mse_naive)
    naive_r2_var = sum([(x - naive_r2_mean) ** 2 for x in r2_naive]) / \
        len(mse_naive)
    naive_mae_var = sum([(x - naive_mae_mean) ** 2 for x in mae_naive]) / \
        len(mse_naive)

    naive_mse_std = naive_mse_var**0.5
    naive_r2_std = naive_r2_var**0.5
    naive_mae_std = naive_mae_var**0.5

    print('### naive mse: ', naive_mse_mean)
    print('### naive mse std: ' + str(naive_mse_std) + '\n')

    print('### naive mae: ', naive_mae_mean)
    print('### naive mae std: ' + str(naive_mae_std) + '\n')

    print('### naive r2: ', naive_r2_mean)
    print('### naive r2 std: ' + str(naive_r2_std) + '\n')