from sklearn.naive_bayes import GaussianNB

import util


def perform_nb(X_train, y_train, X_test, y_test):
    """
    Using Gaussian Naive Bayes, finds best classifier, predicts on test set and calculates confusion
    matrix and other metrics
    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Testing data
    :param y_test: Testing labels
    :return: None
    """

    gnb_clf = GaussianNB()
    gnb_clf.fit(X_train, y_train)
    y_pred = gnb_clf.predict(X_test)

    util.calculate_binary_metrics(y_test, y_pred, metric_of_interest=None, print_mess=True)