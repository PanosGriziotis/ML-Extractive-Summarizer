import pickle
import pandas as pd
import os
from sklearn.svm import SVR
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from data_handling import extract_data_from_json, convert_to_dataset
import os
import argparse


def train_svr_model (X_train: pd.DataFrame, y_train: pd.DataFrame, X_dev: pd.DataFrame,  y_dev: pd.DataFrame, save_dir: str = None):
    """
    Train and save a new svr model from given train data. Find best parameters (C,  kernel type, gamma) for the svr model using GridSearch algorithm and given dev data.
    """

    X = pd.concat([X_train, X_dev],axis=0,ignore_index=True)
    y = pd.concat([y_train, y_dev],axis=0,ignore_index=True)

    
    split_index = [-1 if x in X_train.index else 0 for x in X.index] # Create a list where train data indices are - 1 and validation data indices are 0
    
    pds = PredefinedSplit(test_fold = split_index) # Use the list to create PredefinedSplit
    
    param_grid = [{'C': [1, 10, 100], 'kernel': ['linear']},{'C': [1, 10, 100],'gamma': [0.001, 0.0001],'kernel': ['rbf']}] # define params
    
    regr = SVR()
    best_model = GridSearchCV(estimator = regr, cv = pds, param_grid = param_grid) # Use GridSearchCV
        
    model = best_model.fit(X,y.values.ravel())
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    pickle.dump(model, open(save_dir + '/model.svr', 'wb'))

    print (f'\nTuned on data. Best parameters:\t{best_model.best_params_}\n\n saved in {save_dir}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train svr model using newsroom data files')

    parser.add_argument(
        '--train_file',
        type = str,
        required=True,
        metavar = '',
        help='train data file path'
        )
    parser.add_argument(
        '--dev',
        type=str,
        default=None,
        metavar='',
        help = 'developement data file path'
        )
    parser.add_argument(
        '--num',
        type = int,
        default = 100 ,
        metavar='',
        help = 'number of text-summary pairs to include in train set'
        )
    parser.add_argument(  
        '--split',
        type=int,
        default=10,
        metavar='',
        help = 'proportion of dev data'
        )
    parser.add_argument(
        '-o',
        type=str,
        default='./models',
        metavar='',
        help = 'output folder directory to save trained model'
        )

    args = parser.parse_args()
    
    # model.py [-h] --train_file  [--dev] [--num] [--split] [-o]

    print ('\nTraining model ...')

    train_data, dev_data = extract_data_from_json(data_file = args.train_file, dev_data_file=args.dev, split_ratio = args.split, max_num_of_entries = args.num)

    train_x, train_y = convert_to_dataset(train_data)

    dev_x, dev_y = convert_to_dataset(dev_data)
  
    train_svr_model(train_x, train_y, dev_x, dev_y, args.o)

    print ('\nDone!')

