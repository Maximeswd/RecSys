"""
Codes for preprocessing datasets used in the real-world experiments
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback"
Extended with the code from a "Reproducibility study: Unbiased Pairwise Learning from Biased Implicit Feedback"
"""
import codecs
from pathlib import Path
import itertools
import csv

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from scipy.stats import beta, binom

def transform_rating(ratings: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """ Transform ratings into graded relevance information."""
    ratings = ratings - 1
    return np.round(eps + (1. - eps) * (2 ** ratings - 1) / (2 ** np.max(ratings) - 1))

# Define the data
def mm_est(x, n):
    """ Estimate the parameters of a beta distribution from a binomial sample using the method of moments."""
    
    # Check if x and n are arrays, otherwise convert them to arrays
    if type(x) != np.ndarray:
        x = np.array(x)
    if type(n) != np.ndarray:
        n = np.array(n)
        
    # Initialize starting estimates of true binomial means as raw proportions
    p_i = x / n
    
    #Initialize weights as sample sizes and create a number to store it
    w_i = n
        
    # diff = difference, keep track of how sum of squared difference between new and old weights. 
    diff = 10000
    iterations = 0
    
    while diff > 10 ** (-6):
        
        iterations = iterations + 1
        
        # w is the sum of weights
        w = sum(w_i)
        
        # p is the estimated mean of the underlying beta distribution
        p = sum(w_i * p_i) / w
        
        # s is a weighted estimate of the second moment
        s = sum(w_i * (p_i - p) ** 2)
        
        # phi is the estimated dispersion parameter of the underlying beta distribution
        phi = (s - p * (1 - p) * sum(w_i / n * (1 - w_i / w))) / (p * (1 - p) * (sum(w_i * (1 - w_i / w)) - sum(w_i / n * (1 - w_i / w))))
            
        # Re-calculate weights and squared sum difference between new and old weights
        w_i_old = w_i
        
        w_i = n / (1 + phi * (n - 1))
        
        diff = sum((w_i_old - w_i) ** 2)
        
        # If iterations gets too high, stop and throw an error
        if iterations > 1000: 
            ValueError("Too many iterations, can't converge")
    
    # Convert answers to the traditional alpha, beta format
    alpha = p * (1 - phi) / phi
    beta = (1 - p) * (1 - phi) / phi
    
    return alpha, beta


def bayesian_BB(data: np.ndarray, num_users: int, num_items: int, kind: str):
    
    """
    Emperical Bayesian Estimation of Beta-Binomial (BB) for implicit feedback data.
    """
    
    # Save the train_df as a global variable
    global train_df
    
    # Put train into dataframe with columns user, item, rate and make all values integers
    train_df = pd.DataFrame(data, columns=['user', 'item', 'click']).astype(int)

    if kind == 'user_est':
        
        # Get the number of total clicks per user
        num_clicks_user = train_df.groupby('user')['click'].sum().values
        
        # Each user, could've clicked on each item, so we get the total number of impressions per user
        num_impres_user = np.full(num_users, num_items)  
        
        # Get the estimates of the beta distribution parameters over all users
        alpha, beta = mm_est(num_clicks_user, num_impres_user)
        
        # Get the estimates of the beta distribution parameters for each user
        BB_estimates = []
        for y, n in zip(num_clicks_user, num_impres_user):
            theta = (y + alpha) / (n + alpha + beta)
            BB_estimates.append(theta)

        # Normalize the probability estimates
        BB_estimates = np.array(BB_estimates) / np.sum(BB_estimates)
    
        # return alpha, beta, BB_estimates
        return BB_estimates

    elif kind == 'item_est':
        
        # Get the number of total clicks per user
        num_clicks_item = train_df.groupby('item')['click'].sum().values
        
        # Each user, could've clicked on each item, so we get the total number of impressions per user
        num_impres_item = np.full(num_items, num_users)  
        
        # Get the estimates of the beta distribution parameters over all users
        alpha, beta = mm_est(num_clicks_item, num_impres_item)
        
        # Get the estimates of the beta distribution parameters for each user
        BB_estimates = []
        for y, n in zip(num_clicks_item, num_impres_item):
            theta = (y + alpha) / (n + alpha + beta)
            BB_estimates.append(theta)

        # Normalize the probability estimates
        BB_estimates = np.array(BB_estimates) / np.sum(BB_estimates)   
        
        return BB_estimates

    elif kind == 'combi':
        
        # Get the number of total clicks per user
        num_clicks_user = train_df.groupby('user')['click'].sum().values
        
        # Get the number of total clicks per user
        num_clicks_item = train_df.groupby('item')['click'].sum().values
        
        # Each user, could've clicked on each item, so we get the total number of impressions per user
        num_impres_user = np.full(num_users, num_items)  
        
        # Each user, could've clicked on each item, so we get the total number of impressions per user
        num_impres_item = np.full(num_items, num_users)  
        
        # Get the estimates of the beta distribution parameters over all users
        alpha_user, beta_user = mm_est(num_clicks_user, num_impres_user)
        alpha_item, beta_item = mm_est(num_clicks_item, num_impres_item)
        
        # Get the estimates of the beta distribution parameters for each user
        BB_estimates_user = []
        BB_estimates_item = []
        for y, n in zip(num_clicks_user, num_impres_user):
            theta = (y + alpha_user) / (n + alpha_user + beta_user)
            BB_estimates_user.append(theta)

        for y, n in zip(num_clicks_item, num_impres_item):
            theta = (y + alpha_item) / (n + alpha_item + beta_item)
            BB_estimates_item.append(theta)
            
        # loop through all user item pairs and get the combined estimate
        BB_estimates_combi = []
        for user in range(num_users):
            for item in range(num_items):
                BB_estimates_combi.append((0.55 * BB_estimates_user[user]) * (0.45 * BB_estimates_item[item]))
                
        # Normalize the probability estimates
        BB_estimates_combi = np.array(BB_estimates_combi) / np.sum(BB_estimates_combi)
        
        # Delete the dataframe to save memory
        train_df['theta'] = BB_estimates_combi
        
        return BB_estimates_combi

def preprocess_dataset(data: str, propensity: str):
    """Load and preprocess datasets."""
    np.random.seed(12345)
    if data == 'yahoo':
        cols = {0: 'user', 1: 'item', 2: 'rate'}
        with codecs.open(f'../data/yahoo/raw/train.txt', 'r', 'utf-8', errors='ignore') as f:
            train_ = pd.read_csv(f, delimiter='\t', header=None)
            train_.rename(columns=cols, inplace=True)
        with codecs.open(f'../data/yahoo/raw/test.txt', 'r', 'utf-8', errors='ignore') as f:
            test_ = pd.read_csv(f, delimiter='\t', header=None)
            test_.rename(columns=cols, inplace=True)
        for _data in [train_, test_]:
            _data.user, _data.item = _data.user - 1, _data.item - 1
    elif data == 'coat':
        cols = {'level_0': 'user', 'level_1': 'item', 2: 'rate', 0: 'rate'}
        with codecs.open(f'../data/coat/raw/train.ascii', 'r', 'utf-8', errors='ignore') as f:
            train_ = pd.read_csv(f, delimiter=' ', header=None)
            train_ = train_.stack().reset_index().rename(columns=cols)
            train_ = train_[train_.rate != 0].reset_index(drop=True)
        with codecs.open(f'../data/coat/raw/test.ascii', 'r', 'utf-8', errors='ignore') as f:
            test_ = pd.read_csv(f, delimiter=' ', header=None)
            test_ = test_.stack().reset_index().rename(columns=cols)
            test_ = test_[test_.rate != 0].reset_index(drop=True)
    # count the num. of users and items.
    num_users, num_items = train_.user.max() + 1, train_.item.max() + 1
    
    # Preprocessing according to the paper
    train, test = train_.values, test_.values

    # transform rating into (0,1)-scale.
    test[:, 2] = transform_rating(ratings=test[:, 2], eps=0.0)
    rel_train = np.random.binomial(n=1, p=transform_rating(ratings=train[:, 2], eps=0.1))

    # extract only positive (relevant) user-item pairs
    train = train[rel_train == 1, :2]

    # preprocess for ngcf
    ngcf_train, ngcf_test = _ngcf(train)

    # Creating training data
    all_data = pd.DataFrame(np.zeros((num_users, num_items))).stack().reset_index().values[:, :2]
    unlabeled_data = np.array(list(set(map(tuple, all_data)) - set(map(tuple, train))), dtype=int)
    train = np.r_[np.c_[train, np.ones(train.shape[0])], np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0])]]

    # Estimate propensities and user-item frequencies.
    if data == 'yahoo':
        if propensity == 'bb-item':
            user_freq = np.unique(train[train[:, 2] == 1, 0], return_counts=True)[1] # this returns the total number of clicks per user, len = 15229 (which should be 15400)
            item_freq = np.unique(train[train[:, 2] == 1, 1], return_counts=True)[1]
            pscore = bayesian_BB(train, num_users, num_items, kind='item_est')
            nscore = 1 - pscore
        elif propensity == 'bb-item-user':
            user_freq = np.unique(train[train[:, 2] == 1, 0], return_counts=True)[1] # this returns the total number of clicks per user, len = 15229 (which should be 15400)
            item_freq = np.unique(train[train[:, 2] == 1, 1], return_counts=True)[1]
            pscore = bayesian_BB(train, num_users, num_items, kind='combi')
            nscore = 1 - pscore
        elif propensity == 'original':
            user_freq = np.unique(train[train[:, 2] == 1, 0], return_counts=True)[1] # this returns len = 15229 (which should be 15400 I would say)
            item_freq = np.unique(train[train[:, 2] == 1, 1], return_counts=True)[1]
            pscore = (item_freq / item_freq.max()) ** 0.5
            nscore = 1 - pscore
        else:
            raise ValueError(f"Invalid propensity method: {propensity}")
    
    elif data == 'coat':
        if propensity == 'bb-item':
            pscore = bayesian_BB(train, num_users, num_items, kind='item_est')
            nscore = 1 - pscore
        elif propensity == 'bb-item-user':
            pscore = bayesian_BB(train, num_users, num_items, kind='combi')
            nscore = 1 - pscore
        elif propensity == 'original':
            matrix = sparse.lil_matrix((num_users, num_items))
            for (u, i, y) in train:
                matrix[u, i] = y
            pscore = np.clip(np.array(matrix.mean(axis=0)).flatten() ** 0.5, a_max=1.0, a_min=1e-6)
            nscore = 1 - pscore
        else:
            raise ValueError(f"Invalid propensity method: {propensity}")
        
    # train-val split using the raw training datasets
    train, val = train_test_split(train, test_size=0.1, random_state=12345)
    # save preprocessed datasets
    path_data = Path(f'../data/{data}/{propensity}')
    (path_data / 'point').mkdir(parents=True, exist_ok=True)
    (path_data / 'pair').mkdir(parents=True, exist_ok=True)
    (path_data / 'ngcf').mkdir(parents=True, exist_ok=True)
    
    # pointwise
    np.save(file=path_data / 'point/train.npy', arr=train.astype(int))
    np.save(file=path_data / 'point/val.npy', arr=val.astype(int))
    np.save(file=path_data / 'point/test.npy', arr=test)
    np.save(file=path_data / 'point/pscore.npy', arr=pscore)
    np.save(file=path_data / 'point/nscore.npy', arr=nscore)
    if data == 'yahoo':
        np.save(file=path_data / 'point/user_freq.npy', arr=user_freq)
        np.save(file=path_data / 'point/item_freq.npy', arr=item_freq)
    # pairwise
    samples = 10
    bpr_train = _bpr(data=train, n_samples=samples)
    ubpr_train = _ubpr(data=train, pscore=pscore, n_samples=samples, propensity=propensity)
    bpr_val = _bpr(data=val, n_samples=samples)
    ubpr_val = _ubpr(data=val, pscore=pscore, n_samples=samples, propensity=propensity)
    pair_test = _bpr_test(data=test, n_samples=samples)

    #ngcf 
    ngcf_train.to_csv(path_data / 'ngcf/train.txt', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar='')
    ngcf_test.to_csv(path_data / 'ngcf/test.txt', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar='')


    # New model 
    dubpr_train = _dubpr(data=train, n_samples=samples, pscore=pscore, nscore=nscore, propensity=propensity)
    dubpr_val = _dubpr(data=val, n_samples=samples, pscore=pscore, nscore=nscore, propensity=propensity) 

    np.save(file=path_data / 'pair/bpr_train.npy', arr=bpr_train)
    np.save(file=path_data / 'pair/ubpr_train.npy', arr=ubpr_train)
    np.save(file=path_data / 'pair/bpr_val.npy', arr=bpr_val)
    np.save(file=path_data / 'pair/ubpr_val.npy', arr=ubpr_val)
    np.save(file=path_data / 'pair/test.npy', arr=pair_test)

    # Save new model
    np.save(file=path_data / 'pair/dubpr_train.npy', arr=dubpr_train)
    np.save(file=path_data / 'pair/dubpr_val.npy', arr=dubpr_val)


def _ngcf(data: np.ndarray) -> Tuple:
    def group_items_by_user(data: np.ndarray) -> pd.Series:
        df = pd.DataFrame(data, columns = ['user','item'])
        df = df.groupby('user').agg({'item': lambda x: ' '.join(map(str, x))}).reset_index()
        df['row'] = df['user'].astype(str) + ' ' + df['item']
        return df['row']
    train, test = train_test_split(data, test_size=0.1, random_state=12345)
    return group_items_by_user(train), group_items_by_user(test)


def _bpr(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the naive bpr."""
    df = pd.DataFrame(data, columns=['user', 'item', 'click'])
    positive = df.query("click == 1")
    negative = df.query("click == 0")
    ret = positive.merge(negative, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)

    return ret[['user', 'item_x', 'item_y']].values


def _bpr_test(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Generate training data for the naive bpr."""
    df = pd.DataFrame(data, columns=['user', 'item', 'gamma'])
    ret = df.merge(df, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)

    return ret[['user', 'item_x', 'item_y', 'gamma_x', 'gamma_y']].values


def _ubpr(data: np.ndarray, pscore: np.ndarray, n_samples: int, propensity: str) -> np.ndarray:
    """Generate training data for the unbiased bpr."""

    if propensity == 'bb-item-user':
    # Put the data into a dataframe
        data2 = pd.DataFrame(data, columns=['user', 'item', 'click'])

        # Only select the rows of df that match the rows of data2 
        df = pd.merge(train_df, data2, on=['user', 'item', 'click'], how='inner')

    elif propensity in ['bb-item', 'original']:
        data = np.c_[data, pscore[data[:, 1].astype(int)]]
        df = pd.DataFrame(data, columns=['user', 'item', 'click', 'theta'])

    else:
        raise ValueError(f"Invalid propensity method: {propensity}")
        
    positive = df.query("click == 1")
    ret = positive.merge(df, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)
    ret = ret[ret["item_x"] != ret["item_y"]]

    return ret[['user', 'item_x', 'item_y', 'click_y', 'theta_x', 'theta_y']].values



def _dubpr(data: np.ndarray, pscore: np.ndarray, nscore: np.ndarray, n_samples: int, propensity: str) -> np.ndarray:
    """Generate training data for the dual unbiased bpr."""
    
    if propensity == 'bb-item-user':
        # Put the data into a dataframe
        data2 = pd.DataFrame(data, columns=['user', 'item', 'click'])

        # Only select the rows of df that match the rows of data2 
        train_df['theta_n'] = nscore
        df = pd.merge(train_df, data2, on=['user', 'item', 'click'], how='inner')
        df.columns = ['user', 'item', 'click', 'theta_p', 'theta_n']
    
    elif propensity == 'bb-item':
        data = np.c_[data, np.array(pscore)[data[:, 1].astype(int)], np.array(nscore)[data[:, 1].astype(int)]]
        df = pd.DataFrame(data, columns=['user', 'item', 'click', 'theta_p', 'theta_n'])
    
    elif propensity == 'original':
        data = np.c_[data, np.array(pscore)[data[:, 1].astype(int)], np.array(nscore)[data[:, 1].astype(int)]]
        df = pd.DataFrame(data, columns=['user', 'item', 'click', 'theta_p', 'theta_n'])
    
    else:
        raise ValueError(f"Invalid propensity method: {propensity}")
    
    
    positive = df.query("click == 1")

    ret = positive.merge(df, on="user")\
        .sample(frac=1, random_state=12345)\
        .groupby(["user", "item_x"])\
        .head(n_samples)
    ret = ret[ret["item_x"] != ret["item_y"]]

    return ret[['user', 'item_x', 'item_y', 'click_y', 'theta_p_x', 'theta_p_y', 'theta_n_x', 'theta_n_y']].values

if __name__ == "__main__":
    
    data = 'coat'
    propensity = 'original'
    preprocess_dataset(data=data, propensity=propensity)

    print('\n', '=' * 25, '\n')
    print(f'Finished Preprocessing {data}!')
    print('\n', '=' * 25, '\n')
