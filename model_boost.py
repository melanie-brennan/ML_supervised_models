import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV

import util
import util_plot


def use_boost(X_train, y_train, X_test, y_test):
    """
    Using ada boost, finds best classifier, creates learning and ROC curves with best classifier
    Also uses the best classifier to  predict labels and gets the confusion matrix and other metrics
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Testing data
    :param y_test: Testing labels
    :return: None
    """

    # Use a grid search to find the best decision tree for the training data
    boost_clf = find_best_boost(X_train, y_train)

    # Create a learning curve using the best decision tree classifier
    util.create_learning_curve(X_train, X_test, y_train, y_test, classifier=boost_clf, classifier_name="Ada Boost", metric_name="error_rate", num_proportions = 10, num_runs=5, subtitle="Mean of 5 runs with shaded standard deviation")

    # Train the best classifier
    boost_clf.fit(X_train, y_train)

    # Use the trained best classifier to predict labels and then view various metrics
    y_pred = boost_clf.predict(X_test)
    util.calculate_binary_metrics(y_test, y_pred, metric_of_interest=None, print_mess=True)

    # Use the trained best classifier to predict the probabilities.
    # The second column is the probability if the positive example
    y_pred_prob = boost_clf.predict_proba(X_test)[:, 1]
    # get the false positive rates, true positive rate, thresholds and Area Under Curve.
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred_prob)
    roc_auc = auc(x=fpr, y=tpr)

    # Plot ROC
    util_plot.plot_roc_curve("Ada Boost Classifier", fpr, tpr, roc_auc, label=None, color="green")


def find_best_boost(X_train, y_train, print_mess=True):
    """
    Uses gridSearchCV to perform an exhaustive search over multiple parameters
    :param X_train: Training data
    :param y_train: Training labels
    :param print_mess: Boolean indicating whether the best results should be printed
    :return: Best AdaBoost classifier
    """

    # Define parameters then place them in a dictionary
    weak_clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)

    lr_options = [0.01, 0.1, 1.0]
    num_estimator_options = [50, 100, 150, 200, 250]

    param_grid = dict(learning_rate=lr_options,
                      n_estimators=num_estimator_options)
    # Set number of cross validation curves (default is None which is 5 folds)
    cv = None
    # Set the scoring method, eg. accuracy, roc_auc, f1
    score_method = "roc_auc"

    # Define a weak classifier to be used in AdaBoost ensemble
    weak_clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)
    ada_clf = AdaBoostClassifier(base_estimator=weak_clf)

    # Use GridSearch to find the best classifier
    cv_grid = GridSearchCV(ada_clf, param_grid, cv=cv, scoring=score_method)
    cv_grid.fit(X_train, y_train)

    # Print the best
    if print_mess:
        print("CVGrid best score:", np.around(cv_grid.best_score_, 3))
        print("CVGrid best estimator, ", cv_grid.best_estimator_)
        print("CVGrid best params ", cv_grid.best_params_)
        print("CVGrid best index: ", cv_grid.best_index_)

    return cv_grid.best_estimator_