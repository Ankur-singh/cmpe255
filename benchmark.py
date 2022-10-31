import sys
import logging
import argparse
from model import Regressor, Classifier
from sklearn.datasets import load_breast_cancer, load_diabetes, load_boston, make_classification, make_regression
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG)

def get_objects(args):
    if args['problem'] == 'classification':
        model = Classifier(verbose=0,ignore_warnings=True, custom_metric=None)
        if args['dataset'] == 'breast_cancer': 
            data = load_breast_cancer()
        elif args['dataset'] == 'artificial':
            data = make_classification(n_samples=args['dims'][0], n_features=args['dims'][1])
        else:
            raise ValueError('Invalid dataset name')

    elif args['problem'] == 'regression':
        model = Regressor(verbose=0, ignore_warnings=True, custom_metric=None)
        if args['dataset'] == 'boston':
            data = load_boston()
        elif args['dataset'] == 'diabetes':
            data = load_diabetes()
        elif args['dataset'] == 'artificial':
            data = make_regression(n_samples=args['dims'][0], n_features=args['dims'][1])
        else:
            raise ValueError('Invalid dataset name')

    else:
        logging.info("For usage run : python benchmark.py -h")
        sys.exit()
    return model, data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', choices=['classification', 'regression'], help='Select dataset: classification or regression', required=True)
    parser.add_argument('--dataset', choices=['breast_cancer', 'diabetes', 'boston', 'artificial'], help='selection dataset: breast_cancer, diabetes, boston or artificial', required=True)
    parser.add_argument('--dims', type=int, nargs='+', help='Artificial dataset dimensions', required=False)
    parser.add_argument('--prefix', default='py39', help='which runtime environment? Default: py39', required=False)
    parser.add_argument('--save', help='output filename', required=False)

    args = vars(parser.parse_args())
    
    print(args)
    model, data = get_objects(args)
    fname = f"{args['prefix']}_{args['save']}" if args["save"] else f"{args['prefix']}_{args['dataset']}.csv"
    fname = f"results/{fname}"
    if isinstance(data, tuple):
        X, y = data
    else:
        X = data.data
        y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state =123)
    models,predictions = model.fit(X_train, X_test, y_train, y_test)
    models.to_csv(fname)
    logging.info(f"Done . . . {args['dataset']}. Saved at {fname}")