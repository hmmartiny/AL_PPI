#!/usr/bin/python3

import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import pickle
import random
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-X', '--Xfile',
        help='File with X data',
        type=str,
        default=None,
        dest='X_filename'
    )

    parser.add_argument(
        '-y', '--yfile',
        help='File with y data',
        type=str,
        default=None,
        dest='y_filename'
    )

    parser.add_argument(
        '-an', '--add_n',
        type=int,
        default=1,
        help='Add n samples from pool per training loop (default: 1)',
        dest='add_n'
    )

    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.7,
        help='Desired threshold for accuracy on test set (default: 0.7)',
        dest='threshold'
    )
    
    return parser.parse_args()

def train_test_split_rand(X, y, test_size=0.2, seed = None):
    """Split a dataset into training and test at random

    Parameters
    ----------
    X : pd.DataFrame
        Features of the dataset
    y : pd.DataFrame
        Labels of the dataset
    test_size : float, optional
        Percentage of data in test set, by default 0.2
    seed : int, optional
        Seed for Random State, by default None

    Returns
    -------
    X_train, X_test, y_train, y_test
        Splitted data
    """

    X = X.values
    y = y.values
    
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        
    return X_train, X_test, y_train, y_test

def train_test_split_balanced(X, y, test_size=0.2, seed = None):
    """Split a dataset into training and test while preserving the distribution of y labels

    Parameters
    ----------
    X : pd.DataFrame
        Features of the dataset
    y : pd.DataFrame
        Labels of the dataset
    test_size : float, optional
        Percentage of data in test set, by default 0.2
    seed : int, optional
        Seed for Random State, by default None

    Returns
    -------
    X_train, X_test, y_train, y_test
        Splitted data
    """
    # split into training and test sets while preserving label distribution
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)

    X = X.values
    y = y.values
    
    # split
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        
    return X_train, X_test, y_train, y_test


def train_test_split_pos(X, y, pos_frac=0.2, test_size=0.2):
    """Sample frac percentage of positive samples to be in training data

    Parameters
    ----------
    X : pd.DataFrame
        Features of the dataset
    y : pd.DataFrame
        Labels of the dataset
    pos_frac : float, optional
        Number of positive samples to be in the training set, by default 0.2
    test_size : float, optional
        Percentage of data in the test set, by default 0.2

    Returns
    -------
    X_train, X_test, y_train, y_test
        Splitted data
    """

    # sample 20 % of positive samples to be in training  pool
    y_pos = y.loc[y.y == 1].sample(frac=pos_frac) 
    X_pos = X.loc[y_pos.index]
    
    # figure out whats to be in training data
    train_size = 1 - test_size
    train_n = np.ceil(X.shape[0] * train_size)
    missing_n = int(train_n - X_pos.shape[0])
    
    # sample missing_n from negative samples
    y_neg = y.loc[y.y == 0].sample(missing_n) 
    X_neg = X.loc[y_neg.index]
    
    # combine training data
    X_train = pd.concat([
        X_pos,
        X_neg
    ], axis=0)
    y_train = pd.concat([
        y_pos,
        y_neg
    ], axis=0)
    
    # create test data
    X_test = X.drop(index=X_train.index)
    y_test = y.drop(index=X_train.index)
    
    return X_train.values, X_test.values, y_train.values, y_test.values


def calc_performance(y_true, y_pred, prefix=''):
    """Calculate performance metrics between predicted labels and true labels

    Parameters
    ----------
    y_true : [type]
        [description]
    y_pred : [type]
        [description]
    prefix : str, optional
        [description], by default ''

    Returns
    -------
    [type]
        [description]
    """
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred, zero_division=0)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    
    print(f"{prefix} Accuracy: {acc:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}, F1-score: {f1:.3f}")

    return acc, recall, precision, f1


def uncertainty_sampling(X_pool, y_pool, X_test, y_test, model, model_args, add_n=1, n_init=10, steps=None, verbose=0, threshold=0.7):
    """Active Learning system with pool based sampling by uncertainty

    Parameters
    ----------
    X_pool : numpy.array
        Features in pool
    y_pool : numpy.array
        Labels in pool
    X_test : numpy.array
        Features in test
    y_test : numpy.array
        Labels in test
    model : sklearn.model
        Sklearns implementation of a model, either RandomForestClassifier or MLPClassifier
    model_args : dict
        Argumnts for model function
    add_n : int, optional
        How many samples to add at each iteration, by default 1
    n_init : int, optional
        How many samples from pool are in the training data in the first iteration, by default 10
    steps : int, optional
        Number of iterations to run. If None, then run until all samples from pool have been used or if threshold given, stop when threshold is reached, by default None
    verbose : int, optional
        If 1, print test performance in each iteration, by default 0
    threshold : float, optional
        If given, stop learning system if accuracy score on test set is >= threshold, by default 0.7

    Returns
    -------
    clf : sklearn.model
        Trained classifier
    test_acc, test_recall, test_precision, test_f1: lists
        Performance measures on test set at each iteration
    """

    # mix up order of pool indexes
    order = np.random.permutation(range(len(X_pool)))

    # initialize poolidxs
    poolidxs = np.arange(len(X_pool))

    # take n_init samples from pool as training set
    trainset = order[:n_init]
    X_train = X_pool[trainset]
    y_train = y_pool[trainset]

    # remove the first n_init idxs from poolidxs
    poolidxs = np.setdiff1d(poolidxs, trainset)

    # initialize model
    clf = model(**model_args)

    if steps is None:
        steps = len(poolidxs) // add_n
    
    # training loop
    test_acc, test_recall, test_precision, test_f1 = [], [], [], []
    for i in range(steps):

        # fit model
        clf.fit(X_train, y_train.ravel())

        # calculate performance on test set
        y_pred = clf.predict(X_test)
        acc, recall, precision, f1 = calc_performance(y_true=y_test, y_pred=y_pred)
        test_acc.append((len(X_train), acc))
        test_recall.append((len(X_train), recall))
        test_precision.append((len(X_train), precision))
        test_f1.append((len(X_train), f1))
        

        # calculate label probabilities for samples remaining in pool
        y_prob = clf.predict_proba(X_pool[poolidxs])

        # sort probabilities by least confident max label and sort in negative order
        # x* = argmax (1-p_model(y(1)|x)) for x in unlabeled sample pool
        y_prob_idx_sorted = np.argsort(-y_prob.max(1))

        # add least confident sample to training data
        lc_idx = y_prob_idx_sorted[-add_n:]
        new_idx = poolidxs[lc_idx]

        X_add = X_pool[new_idx]
        y_add = y_pool[new_idx]

        X_train = np.concatenate((
            X_train,
            X_add
        ))
        y_train = np.concatenate((
            y_train,
            y_add
        ))

        # remove from pool
        poolidxs = np.setdiff1d(poolidxs, new_idx)

        if verbose == 1:
            print(f"Step {i+1}/{steps}: Test accuracy: {acc:.3f}", end='\r')

        if threshold is not None and acc >= threshold:
            print("Desired accuracy reached. Stopping training.")
            break
    
    return clf, test_acc, test_recall, test_precision, test_f1


def train(X, y, split_func, sampling_func, add_n, steps=12, model=RandomForestClassifier, model_args={}, split_args={}):
    """Wrapper function for setting up the active learning system"""

    X_train, X_test, y_train, y_test = split_func(X, y, **split_args)

    _, test_acc, test_recall, test_precision, test_f1 = uncertainty_sampling(
        X_pool = X_train,
        X_test = X_test,
        y_pool = y_train,
        y_test = y_test,
        model = model,
        model_args=model_args,
        verbose=1,
        add_n=add_n,
        n_init=add_n,
        threshold=None,
        steps=12
    )

    return test_acc, test_recall, test_precision, test_f1 


if __name__ == "__main__":
    args = parse_args()
    
    # load files
    X = pd.read_csv(args.X_filename, index_col=[0, 1])
    y = pd.read_csv(args.y_filename,  index_col=[0, 1])

    # parameter settings for classifiers
    models = {
        'Random Forest': {
            'model': RandomForestClassifier,
            'model_args': {
                'n_estimators': 20, 
                'min_samples_split': 7, 
                'n_jobs': -1
            }
        }, 
        'Neural Network': {
            'model': MLPClassifier,
            'model_args': {}
        }
    }


    # functions and settings for data splitters 
    split_funcs = [train_test_split_rand, train_test_split_balanced, train_test_split_pos]
    split_args = {
        'Random': [{}],
        'Stratified': [{}],
        'Random positive sampling': [{'pos_frac': x} for x in [0.01, 0.2, 0.45]],
    }
    split_keys = ['Random', 'Stratified', 'Random positive sampling']
    
    # dict to store results in
    res = defaultdict(lambda: defaultdict(dict))

    # test each combination
    for model_name, model_settings in models.items():
        for split_func, split_key in zip(split_funcs, split_keys):
            for split_arg in split_args[split_key]:
                
                split_label = split_key
                if 'Random positive sampling' == split_key:
                    split_label = split_key.replace('Random', f"{split_arg['pos_frac']*100}%")
                
                for i in range(5): # repeat 5 times
                    print(f"{model_name} {split_label} - iteration: {i+1}/5")

                    # set random seed
                    random.seed()

                    test_acc, test_recall, test_precision, test_f1 = train(
                        X=X, 
                        y=y, 
                        split_func=split_func, 
                        sampling_func=uncertainty_sampling,
                        add_n=args.add_n,
                        model=model_settings['model'],
                        model_args=model_settings['model_args'],
                        split_args=split_arg
                    )

                    res[model_name][split_label][i] ={
                        #'test_acc': test_acc,
                        'test_recall': test_recall,
                        'test_precision': test_precision,
                        'test_f1': test_f1
                    }
    
    # dump result dict in a pickled file
    with open('results.pkl', 'wb') as dest:
        pickle.dump(res, dest)
    