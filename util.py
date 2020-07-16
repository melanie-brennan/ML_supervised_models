import io
import requests
import zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import util_plot

def print_duration(start_time, end_time):
    """
    Input is two times. Calculates the difference in the times with various units
    :param start_time:
    :param end_time:
    :return: None
    """

    seconds = end_time - start_time
    minutes = seconds / 60.0
    hours = minutes / 60.0

    print("\nTime taken with various units:")
    print("Seconds: ", seconds)
    print("Minutes: ", minutes)
    print("Hours:   ", hours)


def download_extract_zip(zip_url):
    """
    Downloads a zip file and extracts it
    :param zip_url: url of the zip file
    :return:
    """
    r = requests.get(zip_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()


def create_train_test_sets(csv_filename, test_proportion, label_column):
    """
    Reads data from csv file then performs a stratified split (based on the label column).
    The proportion of test data is defined by test proportion and the proportion of training data is (1- test_proportion)
    :param csv_filename: filename of csv file
    :param test_proportion: proportion of data that will go into the test set.
    The proportion in the training set will be (1- test_proportion)
    :param label_column: Column that contains the true classification labels
    :return:
    """
    #set seed
    seed = 42

    # Read the dataset into a pandas dataframe
    df = pd.read_csv(csv_filename, sep=",", header=None)

    # Convert to a numpy ndarray
    all_data = df.values

    # Separate the features and the labels
    X = all_data[:, :label_column]
    y = all_data[:, label_column]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_proportion, random_state=seed, stratify=y)
    #print(X_train.shape)
    #print(X_test.shape)
    #print(y_train.shape)
    #print(y_test.shape)

    # Use StandardScaler.  Fit to training data and apply to training and test data
    ss = StandardScaler()
    ss.fit(X_train)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)

    return (X_train, X_test, y_train, y_test)


def calculate_binary_metrics(true_labels, predicted_labels, metric_of_interest=None, print_mess=True):
    """
    Calculates a number of metrics - coinfusion table, accuracy, error rate, precision, recall, f1
    true positive rate, true negative rate, false positive rate, false negative rate
    :param true_labels: Ground truth labels
    :param predicted_labels: Predicted labels
    :param metric_of_interest: Metric that will be returned from the function
    :param print_mess: Boolean indicating wither metrics should be printed
    :return: moi, value of the metric of interest
    """
    num_labels = true_labels.shape[0]

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    [[tn, fp],[fn, tp]] = conf_matrix

    # Convert to floats
    (tn, fn, fp, tp) = (float(tn), float(fn), float(fp), float(tp))

    # Calculate accuracy (normalised) - manually and with sklearn
    accuracy_manual = (tp + tn) / (fp + fn + tp + tn)
    acccuracy_sklearn = accuracy_score(true_labels, predicted_labels)

    # Calculate precison - manually and with sklearn
    precision_manual = tp / (tp + fp)
    precision_sklearn = precision_score(true_labels, predicted_labels)

    # Calculate recall - manually and with sklearn
    recall_manual = tp / (tp + fn)
    recall_sklearn = recall_score(true_labels,predicted_labels)

    # Calculate f1 - manually and with sklearn for confirmation
    f1_manual1 = 2.0 * (precision_manual) * (recall_manual) / (precision_manual + recall_manual)
    f1_manual2 = tp / (tp + ((fn+fp)/2.0))
    f1_sklearn = f1_score(true_labels, predicted_labels)

    # Calculate error rate (misclassification rate)
    error_rate_manual = (fp+fn) / (fp + fn + tp + tn)

    # Calculate true positive rate (sensitivity) true negative rate (specificity)
    tpr = tp /(tp + fn)
    tnr = tn /(tn + fp)

    #Calculate false positive rate and false negative rate
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    if print_mess:
        print("\nNumber of labels to predict: ", num_labels)

        print("\nConfusion matrix...")
        print(conf_matrix)

        print("\nTrue negative: ", tn)
        print("False negative: ", fn)
        print("False positive: ", fp)
        print("True positive: ", tp)

        print("\nNumber correct: ", (tp + tn), ", out of ", (tp + tn + fn + fp))

        print("\nManual calculation of accuracy normalised: ", accuracy_manual)
        print("Sklearn accuracy: ", acccuracy_sklearn)

        print("\nManual calculation of error rate (misclassification rate): ", error_rate_manual)

        print("\nManual calculation of precision: ", precision_manual)
        print("Sklean Precision: ", precision_sklearn)

        print("\nManual calculation of recall: ", recall_manual)
        print("Sklearn recall...", recall_sklearn)

        print("\nManual calculation of f1 (method 1): ", f1_manual1)
        print("Manual calculation of f1 (method 2): ", f1_manual2)
        print("sklearn f1: ", f1_sklearn)

        print("\nTrue positive rate (sensitivity): ", tpr)
        print("True negative rate (specificity): ", tnr)

        print("\nFalse positive rate: ", fpr)
        print("False negative rate: ", fnr)

        print("-----------------------------------------")

    # return metric
    if metric_of_interest == "error_rate":
        moi = error_rate_manual
    elif metric_of_interest == "accuracy":
        moi = acccuracy_sklearn
    elif metric_of_interest == "f1":
        moi = f1_sklearn
    else:
        moi = None

    return moi



def create_learning_curve(X_train, X_test, y_train, y_test, classifier, classifier_name= "", metric_name="", num_proportions = 5, num_runs = 1, subtitle =''):
    """
    :param X_train: Training data
    :param X_test: Testing data
    :param y_train: Training labels
    :param y_test: Testing labels
    :param classifier: classsifier that will be used
    :param classifier_name: String that will display the classifiers name in the plot
    :param metric_name: String that will diplay the metric used.  Appears on plots y axis
    :param num_proportions: Number of proportions that the training data will be progessively split
    :param num_runs: Number of times the splitting procedure should be repeated
    :param subtitle: string that will appear in the second line of the plot title.
    :return: 
    """

    # Get the proportions of training data to be used, default num_proportions is 6, resulting in [0 0.2 0.4 0.6 0.8 1)
    proportions_arr  = np.linspace(0.0, 1.0, num_proportions+1)
    # Discard the first element (which is zero)
    proportions_arr = proportions_arr[1:]

    # Create lists to hold the results
    training_metrics_mean = []
    testing_metrics_mean = []
    training_metrics_sd = []
    testing_metrics_sd = []

    for proportion in proportions_arr:
        # Create lists to hold metrics for each run
        temp_training_metrics = []
        temp_testing_metrics = []

        # When num_runs is 1 (the default) the mean will just be the measuremant and std dev will be 0
        for n in range(num_runs):

            # Calculate how many instances should be used in the training sample
            num_training_samples = int(X_train.shape[0] * proportion)

            # Restrict the number of training samples
            reduced_X_train = X_train[:num_training_samples, :]
            reduced_y_train = y_train[:num_training_samples]

            # Train the model
            classifier.fit(reduced_X_train, reduced_y_train)

            # Make the predictions (note that the predict method uses a threshold of 0.5)
            predicted_train_labels = classifier.predict(reduced_X_train)
            predicted_test_labels = classifier.predict(X_test)

            # Get metric (eg.error)
            temp_training_error = calculate_binary_metrics(reduced_y_train, predicted_train_labels, metric_name, print_mess=False)
            temp_testing_error = calculate_binary_metrics(y_test, predicted_test_labels, metric_name, print_mess=False)

            # Append results to a list
            temp_training_metrics.append(temp_training_error)
            temp_testing_metrics.append(temp_testing_error)

        # Get the mean and standard error for the runs
        training_runs_mean = np.mean(np.asarray(temp_training_metrics))
        testing_runs_mean = np.mean(np.asarray(temp_testing_metrics))
        training_runs_sd = np.std(np.asarray(temp_training_metrics))
        testing_runs_sd = np.std(np.asarray(temp_testing_metrics))

        # Append the metrics to the lists
        training_metrics_mean.append(training_runs_mean)
        testing_metrics_mean.append(testing_runs_mean)
        training_metrics_sd.append(training_runs_sd)
        testing_metrics_sd.append(testing_runs_sd)

    #print(training_metrics_mean)
    #print(testing_metrics_mean)
    #print(training_metrics_sd)
    #print(testing_metrics_sd)

    #plot the learning curve
    util_plot.plot_learning_curve(proportions_arr, training_metrics_mean, testing_metrics_mean, training_metrics_sd, testing_metrics_sd, classifier_name, metric_name, subtitle)