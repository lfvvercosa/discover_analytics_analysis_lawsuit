from scipy.stats.mstats import winsorize
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def normalize(df, cols = None):
    result = df.copy()
    
    if cols == None:
        cols = df.columns

    for feature_name in cols:
        if result[feature_name].max() != result[feature_name].min():
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) \
                / (max_value - min_value)
    
    return result


def pre_process_dataset(df, gt, not_a_feature):
    if 'TOTAL_TRACES' in df.columns:
        df = df[df['TOTAL_TRACES'] != 0]
    
    # df = df[(df[gt] != -1) & (df[gt] != 0)  & (df[gt] != 1)]
    df = df[(df[gt] != -1) & (df[gt] != 0)]
    df = df[df[gt].notna()]

    for c in df.columns:
        if c not in not_a_feature and is_numeric_dtype(df[c]):
            df[c] = winsorize(df[c], (0.05, 0.05))
            df[c].fillna((df[c].mean()), inplace=True)

            if df[c].dtype == np.float64:
                df[c] = df[c].replace(-1, df[c].mean())
        
        if c == 'DISCOVERY_ALG':
            df[c] = pd.Categorical(df[c]).codes
    
    feat = [f for f in df.columns if f not in not_a_feature]
    X = df[feat]
    y = df[not_a_feature]
    X = normalize(X)
    df = X.join(y)

    return df

