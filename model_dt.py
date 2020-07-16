import numpy as np
from sklearn import tree
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV

import util
import util_plot


def use_dt(X_train, y_train, X_test, y_test):
    """
    Using decision trees, finds best classifier, creates learning and ROC curves with best classifier
    Also uses the best classifier to  predict labels and gets the confusion matrix and other metrics
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Testing data
    :param y_test: Testing labels
    :return: None
    """

    # Use a grid search to find the best decision tree for the training data
    dt_clf = find_best_dt(X_train, y_train)

    # Create a learning curve using the best decision tree classifier
    util.create_learning_curve(X_train, X_test, y_train, y_test, classifier=dt_clf, classifier_name="Decision Tree", metric_name="error_rate", num_proportions = 10, num_runs=5, subtitle="Mean of 5 runs with shaded standard deviation")

    # Train the best classifier
    dt_clf.fit(X_train, y_train)

    # Use the trained best classifier to predict labels and then view various metrics
    y_pred = dt_clf.predict(X_test)
    util.calculate_binary_metrics(y_test, y_pred, metric_of_interest=None, print_mess=True)

    # Use the trained best classifier to predict the probabilities.
    # The second column is the probability if the positive example
    y_pred_prob = dt_clf.predict_proba(X_test)[:, 1]
    # get the false positive rates, true positive rate, thresholds and Area Under Curve.
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred_prob)
    roc_auc = auc(x=fpr, y=tpr)
    #Plot ROC
    util_plot.plot_roc_curve("Decision Tree Classifier", fpr, tpr, roc_auc, label=None, color="green")


def find_best_dt(X_train, y_train, print_mess=False):
    """
    Uses GeidSearchCV to find the best decision tree classifier
    :param X_train: Training data
    :param y_train: Training labels
    :param print_mess: Boolean to indicate whether to print the best results
    :return: Best decision tree clasifier
    """

    # Use grid search to perform an exhaustive search over multiple parameters
    # Define parameters then place them in a dictionary
    criterion_options = ['gini', 'entropy']
    max_depth_options = list(range(1, 30))
    splitter_options = ['best', 'random']
    param_grid = dict(criterion=criterion_options,
                      max_depth=max_depth_options,
                      splitter=splitter_options)
    # Set number of cross validation curves (default is None which is 5 folds)
    cv = None

    # Set the scoring method, eg. accuracy, roc_auc, f1
    score_method = "roc_auc"

    # Create classifier
    dt_clf = tree.DecisionTreeClassifier()

    # Use GridSearch to find the best classifier
    cv_grid = GridSearchCV(dt_clf, param_grid, cv=cv, scoring=score_method)
    cv_grid.fit(X_train, y_train)

    # Print the best
    if print_mess:
        print("CVGrid best score:", np.around(cv_grid.best_score_, 3))
        print("CVGrid best estimator, ", cv_grid.best_estimator_)
        print("CVGrid best params ", cv_grid.best_params_)
        print("CVGrid best index: ", cv_grid.best_index_)

    return cv_grid.best_estimator_