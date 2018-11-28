# from sklearn.datasets import load_iris
# from sklearn.pipeline import make_pipeline
# from sklearn import preprocessing
# from sklearn.model_selection import KFold, cross_val_score,cross_val_predict,TimeSeriesSplit
# from sklearn.metrics import accuracy_score
# from sklearn import svm
#
#
# # Load the iris test data
# iris = load_iris()
# # View the iris data features for the first three rows
# print(iris.data[0:10])
# # View the iris data target for first three rows. '0' means it flower is of the setosa species.
# print('target',iris.target[0:10])
# # Create a pipeline that scales the data then trains a support vector classifier
# classifier_pipeline = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
# # KFold/StratifiedKFold cross validation with 3 folds (the default)
# # applying the classifier pipeline to the feature and target data
# scores = cross_val_score(classifier_pipeline, iris.data, iris.target, cv=5,scoring='precision_macro')
# print('scores:',scores)
# print('scores_mean:',scores.mean())
# predicted = cross_val_predict(classifier_pipeline, iris.data, iris.target, cv=5)
# print('scores mean of cross val predict:', accuracy_score(iris.target, predicted))
# tscv = TimeSeriesSplit(n_splits=3).split(iris.data)
# print(tscv)
# score = cross_val_score(classifier_pipeline, iris.data, iris.target, cv=TimeSeriesSplit(n_splits=2).split(iris.data))
from collections import defaultdict

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import TimeSeriesSplit


def cross_validation(
        model,
        params,
        make_feature_pipeline,
        make_target_pipeline,
        pipeline_params,
        train_data,
        n_folds=5,
        verbose=False
):
    """
    Cross validates over a single set of model parameters

    args
    	model (sklearn estimator) uninitialized model
	params (dict)
	make_target_pipeline (function)
	make_target_pipeline (function)
	train_data (pd.DataFrame) fed into both the target and feature pipelines
	n_folds (int)
	verbose (bool)

    TODO ability to use different test_train_splits
    """
    results = defaultdict(list)

    print('running {} folds cross_validation over {}'.format(n_folds, params))
    ts_split = TimeSeriesSplit(n_splits=n_folds)

    for fold, (train_index, test_index) in enumerate(ts_split.split(train_data), 1):
        cv_train = train_data.iloc[train_index, :]
        cv_test = train_data.iloc[test_index, :]

        #  make the training and test data
        x_train = pipeline.fit_transform(cv_train)
        x_test = pipeline.transform(cv_test)

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        cv_model = model(**params)
        cv_model.fit(x_train, y_train.reshape(-1))
        train_score, test_score = 100 * cv_model.score(x_train, y_train), 100 * cv_model.score(x_test, y_test)

        results['train_scores'].append(train_score)
        results['test_scores'].append(test_score)
        results['models'].append(cv_model)
        results['feature_pipe'].append(feature_generator)
        results['target_pipe'].append(target_generator)

        if verbose:
            score_log = 'fold {:.0f} {:.1f} % train score {:.1f} % test score'.format(fold,
                                                                                      train_score,
                                                                                      test_score)
            print(score_log)

    results = dict(results)
    results['params'] = cv_model.get_params()
    results['test_score'] = np.mean(results['train_scores'])
    results['train_score'] = np.mean(results['test_scores'])

    print('CV done - train {} % test {} %'.format(results['train_score'], results['test_score']))

    return results

