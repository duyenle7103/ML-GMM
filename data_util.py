"""data_util.py
This file provides functions to load data for the assignment
Author: Kien Huynh
"""

import numpy as np
import pandas as pd

def get_iris_data(path="./data/iris.dat"):
    """get_iris_data
    
    Load the data into 6 numpy arrays:
    * train_x1
    * train_x2
    * train_x3
    * test_x1
    * test_x2
    * test_x3
    :param path: path to the iris dataset file
    """ 
    dataIris = pd.read_table('data/iris.dat', delim_whitespace = True, header = None)
    dataIris.head()

    dataIris = dataIris.to_numpy()
    
    x1 = dataIris[dataIris[:,4]==1, 0:4]
    x2 = dataIris[dataIris[:,4]==2, 0:4]
    x3 = dataIris[dataIris[:,4]==3, 0:4]
    y1 = dataIris[dataIris[:,4]==1, 4]
    y2 = dataIris[dataIris[:,4]==2, 4]
    y3 = dataIris[dataIris[:,4]==3, 4]

    train_x1 = x1[0:40,:]
    test_x1 = x1[40:,:]
    train_x2 = x2[0:40,:]
    test_x2 = x2[40:,:]
    train_x3 = x3[0:40,:]
    test_x3 = x3[40:,:]

    train_y1 = y1[0:40].reshape((40,1))
    test_y1 = y1[40:].reshape((10,1))
    train_y2 = y2[0:40].reshape((40,1))
    test_y2 = y2[40:].reshape((10,1))
    train_y3 = y3[0:40].reshape((40,1))
    test_y3 = y3[40:].reshape((10,1))

    train_x = np.concatenate((train_x1, train_x2, train_x3), axis=0)
    train_y = np.concatenate((train_y1, train_y2, train_y3), axis=0)
    test_x = np.concatenate((test_x1, test_x2, test_x3), axis=0)
    test_y = np.concatenate((test_y1, test_y2, test_y3), axis=0)

    return (train_x, train_y, test_x, test_y)

def load_iris_data(path="./data/iris_2.data"):
    # Read data from file
    dataframe = pd.read_csv(path, sep=',', header=None)

    # Extract features and labels
    X = dataframe.iloc[:, :-1].values
    Y_str = dataframe.iloc[:, -1].values

    # Create dictionary mapping from string to integer
    label_to_int = {label: idx for idx, label in enumerate(pd.unique(Y_str))}
    Y = np.array([label_to_int[label] for label in Y_str])
    numtypes = len(label_to_int)

    return X, Y, numtypes

def kfold_split(X, Y, n_splits=5, shuffle=True):
    # Get indices
    indices = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(indices)

    # Get integer division
    fold_size = X.shape[0] // n_splits
    for i in range(0, X.shape[0], fold_size):
        test_indices = indices[i:i+fold_size]
        train_indices = np.concatenate((indices[:i], indices[i+fold_size:]))
        yield X[train_indices], Y[train_indices], X[test_indices], Y[test_indices]

def process_input_1():
    # Load data from file
    # Make sure that iris.dat is in data/
    train_x, train_y, test_x, test_y = get_iris_data()
    
    # Change the labels
    train_y = train_y - 1
    test_y = test_y - 1
    train_y_flat = train_y.flatten()
    test_y_flat = test_y.flatten()

    # Calculate the number of classes
    num_classes = (np.unique(train_y)).shape[0]

    return train_x, train_y_flat, test_x, test_y_flat, num_classes