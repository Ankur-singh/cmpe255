import sys
import logging
import argparse
import pandas as pd
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.datasets import load_breast_cancer, load_diabetes, load_boston
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG)

def get_objects(dataset):
    if dataset == 'breast_cancer':
        data = load_breast_cancer()
        model = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    elif dataset == 'diabetes':
        data = load_diabetes()
    elif dataset == 'boston':
        data = load_boston()
    else:
        raise ValueError('Invalid dataset name')
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', choices=['classification', 'regression'], help='Select problem: classification or regression', required=True)
    parser.add_argument('--dataset', choices=['breast_cancer', 'diabetes', 'boston'], help='selection dataset: breast_cancer, diabetes or boston', required=True)
    args = vars(parser.parse_args())
    
    if args['dataset'] == 'breast_cancer':
        model = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
        data = load_breast_cancer()
        fname = 'py37_c_breast.csv'
    else:
            logging.info("select on of the following supported dataset:\n breast_cancer")
            sys.exit()
    elif args['problem'] == 'regression':
        model = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
        if args['dataset'] == 'boston':
            data = load_boston()
            fname = 'py37_c_boston.csv'
        elif args['dataset'] == 'diabetes':
            data = load_diabetes()
            fname = 'py37_c_diabetes.csv'
        else:
            logging.info("select one of the following supported dataset:\n boston, diabetes")
            sys.exit()
    else:
        logging.info("select problem: classification or regression")
        sys.exit()
    
    X = data.data
    y= data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
    models,predictions = model.fit(X_train, X_test, y_train, y_test)
    models.to_csv(fname)
    logging.info(f"Done . . . {args['dataset']}. Saved at {fname}")