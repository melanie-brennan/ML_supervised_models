import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV

import util
import util_plot
import model_dt


def use_rf(X_train, y_train, X_test, y_test):
    """
    Using a random forest, finds best classifier (two methods available), creates learning and ROC curves with best classifier
    Also uses the best classifier to  predict labels and gets the confusion matrix and other metrics
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Testing data
    :param y_test: Testing labels
    :return: None
    """

    # Use GridSearchCV to find the best random forest
    # Note this classifier overfit
    #rf_clf = find_best_rf(X_train, y_train)

    # Alternate method - find the best decision tree using GridSearchCV then use its parameters in RF
    # This decision tree classifier tends to be simpler than the one found by grid erarch above and is less prone to overfitting
    num_estimators = 10
    rf_clf = create_rf_with_best_dt(X_train, y_train, num_estimators)


    # Create a learning curve using the best decision tree classifier
    util.create_learning_curve(X_train, X_test, y_train, y_test, classifier=rf_clf, classifier_name="Random Forest", metric_name="error_rate", num_proportions=10, num_runs=5, subtitle="Mean of 5 runs with shaded standard deviation")

    # Train the best classifier
    rf_clf.fit(X_train, y_train)

    # Use the trained best classifier to predict labels and then view various metrics
    y_pred = rf_clf.predict(X_test)
    util.calculate_binary_metrics(y_test, y_pred, metric_of_interest=None, print_mess=True)

    # Use the trained best classifier to predict the probabilities.
    # The second column is the probability if the positive example
    y_pred_prob = rf_clf.predict_proba(X_test)[:, 1]
    # get the false positive rates, true positive rate, thresholds and Area Under Curve.
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred_prob)
    roc_auc = auc(x=fpr, y=tpr)

    # Plot ROC
    util_plot.plot_roc_curve("Random Forest Classifier", fpr, tpr, roc_auc, label=None, color="green")


def create_rf_with_best_dt(X_train, y_train, num_estimators):
    """
    Finds the best decision tree for the data, then creates a random forest initiated with parameters
    from the best decision tree
    :param X_train: Training data
    :param y_train: Training labels
    :param num_estimators: number of decision trees in the random forest
    :return: random forest classifier initiated that has the best decision trees
    """

    # use a grid search to find the best decision tree for the training data
    best_dt_clf = model_dt.find_best_dt(X_train, y_train)

    # Extract the classifier characteristics
    ccp_alpha = best_dt_clf.ccp_alpha
    class_weight = best_dt_clf.class_weight
    criterion = best_dt_clf.criterion
    max_depth = best_dt_clf.max_depth
    max_features = best_dt_clf.max_features
    max_leaf_nodes = best_dt_clf.max_leaf_nodes
    min_impurity_decrease = best_dt_clf.min_impurity_decrease
    min_impurity_split = best_dt_clf.min_impurity_split
    min_samples_leaf = best_dt_clf.min_samples_leaf
    min_samples_split = best_dt_clf.min_samples_split
    min_weight_fraction_leaf = best_dt_clf.min_weight_fraction_leaf
    random_state = best_dt_clf.random_state

    # Initiate the random forest with parameters from the best decision tree
    rf_clf = RandomForestClassifier(n_estimators= num_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, random_state=random_state,  class_weight=class_weight,ccp_alpha=ccp_alpha)

    return rf_clf


def find_best_rf(X_train, y_train, print_mess=False):
    """
    Uses GridSearchCV to search through different parameter combinations to find the best random forest classifier
    :param X_train: Training data
    :param y_train: Training labels
    :param print_mess: Boolean indicating whether to print best indicators
    :return: best estimator
    """

    # Define parameters to be searched
    param_grid = {
        'n_estimators': [500,1000, 2000],
        'max_depth': [6,8],
        'criterion': ['gini', 'entropy']
    }
    # Set number of cross validation curves (default is None which is 5 folds)
    cv = None
    # Set the scoring method, eg. accuracy, roc_auc, f1
    score_method = "roc_auc"

    # Create classifier and
    rf_clf = RandomForestClassifier()

    # Use GridSearch to find the best classifier
    cv_grid = GridSearchCV(rf_clf, param_grid, cv=cv, scoring=score_method)
    cv_grid.fit(X_train, y_train)

    # Print the best
    if print_mess:
        print("CVGrid best score:", np.around(cv_grid.best_score_, 3))
        print("CVGrid best estimator, ", cv_grid.best_estimator_)
        print("CVGrid best params ", cv_grid.best_params_)
        print("CVGrid best index: ", cv_grid.best_index_)

    return cv_grid.best_estimator_