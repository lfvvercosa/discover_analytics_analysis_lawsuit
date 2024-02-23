from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from models.DatasetSplit import DatasetSplit


def run_linear_regression(df, 
                          splits,
                          random_seed,
                          test_size):

    gt = 'TEMPO_PROCESSUAL_TOTAL_DIAS'
    feat = [c for c in df.columns if c != gt]

    X = df[feat].to_numpy()
    y = df[gt].to_numpy()

    dataset_split = DatasetSplit()
    X_train_, X_test_, y_train_, y_test_ = dataset_split.\
        train_test_split(df, gt, test_size, random_seed)

    # Make target to last column
    cols = list(df.columns)
    cols.remove(gt)
    cols.append(gt)
    df = df[cols]

    X_train_, X_test_, y_train_, y_test_ = dataset_split.\
        train_test_split(df, gt, test_size, random_seed)
    
    skf = dataset_split.kfold(X_train_,
                              y_train_,
                              splits, 
                              random_seed)
                
    mse = []
    r2 = []
    mae = []

    for train_index, test_index in skf:
        # X_trainkf, X_validkf = X_train_[train_index,:-1], X_train_[test_index,:-1]
        # y_trainkf, y_validkf = X_train_[train_index,-1], X_train_[test_index,-1]

        X_trainkf, X_validkf = X_train_[train_index], X_train_[test_index]
        y_trainkf, y_validkf = y_train_[train_index], y_train_[test_index]

        linear_model = LinearRegression()
        linear_model.fit(X_trainkf, y_trainkf)

        y_predkf = linear_model.predict(X_validkf)

        mse.append(mean_squared_error(y_validkf, y_predkf))
        r2.append(r2_score(y_validkf, y_predkf))
        mae.append(mean_absolute_error(y_validkf, y_predkf))

    mse_mean = sum(mse) / len(mse)
    r2_mean = sum(r2) / len(r2)
    mae_mean = sum(mae) / len(mae)

    mse_var = sum([(x - mse_mean) ** 2 for x in mse]) / len(mse)
    r2_var = sum([(x - r2_mean) ** 2 for x in r2]) / len(mse)
    mae_var = sum([(x - mae_mean) ** 2 for x in mae]) / len(mse)

    mse_std = mse_var**0.5
    r2_std = r2_var**0.5
    mae_std = mae_var**0.5

    print('### Error train:')

    print('### lin reg mse: ', mse_mean)
    print('### lin reg mse std: ' + str(mse_std) + '\n')

    print('### lin reg mae: ', mae_mean)
    print('### lin reg mae std: ' + str(mae_std) + '\n')

    print('### lin reg r2: ', r2_mean)
    print('### lin reg r2 std: ' + str(r2_std) + '\n')

    # X_train, y_train = X_train_[:,:-1], X_train_[:,-1]
    # X_test, y_test = X_test_[:,:-1], X_test_[:,-1]

    X_train, X_test = X_train_, X_test_
    y_train, y_test = y_train_, y_test_

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    print('### Error test:')
    print('mean_squared_error: ' + str(mean_squared_error(y_test, y_pred)))
    print('r2_score: ' + str(r2_score(y_test, y_pred)))
    print('mean_absolute_error: ' + str(mean_absolute_error(y_test, y_pred)))