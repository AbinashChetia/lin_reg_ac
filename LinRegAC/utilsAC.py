import numpy as np
import pandas as pd

def splitTrainTest(x, y, train_ratio=0.8):
    '''
    Split data into training and testing sets.
    '''
    df_x = x.copy()
    df_y = y.copy()
    df_y = df_y.rename('y')
    df = pd.concat([df_x, df_y], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    train_size = int(len(df) * train_ratio)
    train_x = df.iloc[:train_size, :-1]
    train_y = df.iloc[:train_size, -1]
    test_x = df.iloc[train_size:, :-1]
    test_y = df.iloc[train_size:, -1]
    return train_x, train_y, test_x, test_y

def splitTrainTest_kfold(x, y, k=5):
    '''
    Split data into training and testing sets for k-fold cross validation.
    '''
    df_x = x.copy()
    df_y = y.copy()
    df_y = df_y.rename('y')
    df = pd.concat([df_x, df_y], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)
    fold_size = int(len(df) / k)
    data_folds = []
    for i in range(k):
        if i != k - 1:
            data_folds.append(df.iloc[i * fold_size: (i + 1) * fold_size, :].reset_index(drop=True))
        else:
            data_folds.append(df.iloc[i * fold_size:, :].reset_index(drop=True))
    return data_folds
    

def normMinMax(df, mode='train', train_min=None, train_max=None):
    '''
    Perform min-max normalization on data.
    '''
    data = df.copy()
    if mode == 'train':
        train_max = {}
        train_min = {}
        for col in data.columns:
            train_max[col] = data[col].max()
            train_min[col] = data[col].min()
            data[col] = (data[col] - train_min[col]) / (train_max[col] - train_min[col])
        return data, train_min, train_max
    
    elif mode == 'test':
        if train_min is None or train_max is None:
            raise Exception('Pass train_min and/or train_max.')
        for col in data.columns:
            data[col] = (data[col] - train_min[col]) / (train_max[col] - train_min[col])
        return data

def getMse(pred, y):
    '''
    Calculate mean squared error.
    '''
    return np.mean(np.square(pred - y))

def getRmse(pred, y):
    '''
    Calculate root mean squared error.
    '''
    return np.sqrt(np.mean(np.square(pred - y)))