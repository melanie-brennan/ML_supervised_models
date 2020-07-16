import os
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(proportions, training_metrics, testing_metrics, training_metrics_sd, testing_metrics_sd, classifier_name = "", metric_name="", subtitle =""):
    """
    Creates a learning curve by limiting the training set from a small proportion to eventually the whole training set
    Plots the training curve and saves it to file
    :param proportions: proportions of the training data used. Will appear on the x-axis
    :param training_metrics: Metric that is the mean of the runs for the training data
    :param testing_metrics: Metric that is the mean of the runs for the testing data
    :param training_metrics_sd: Metric that is the standard deviation of the runs for the testing data
    :param testing_metrics_sd: Metric that is the standard deviation of the runs for the testing data
    :param classifier_name: string of the classifier name that will be displayed in the plot
    :param metric_name: string of the name of the metric used.  It will be displayed along the y axis
    :param subtitle: string that will appear on the second line of the plot
    :return: None
    """

    # Convert from list to ndarray
    training_metrics = np.asarray(training_metrics)
    testing_metrics = np.asarray(testing_metrics)
    training_metrics_sd = np.asarray(training_metrics_sd)
    testing_metrics_sd = np.asarray(testing_metrics_sd)

    # Plot the mean surrounded by shaded area representing standard deviation
    plt.plot(proportions, training_metrics, "red", label=("Training set"))
    plt.fill_between(proportions, training_metrics - training_metrics_sd, training_metrics + training_metrics_sd, color ="red", alpha= 0.2 )
    plt.plot(proportions, testing_metrics, "blue", label=("Test set"))
    plt.fill_between(proportions, testing_metrics - testing_metrics_sd, testing_metrics + testing_metrics_sd, color="blue", alpha=0.2)

    plt.xlabel("Proportion of training data")
    plt.ylabel(metric_name.capitalize().replace("_", " "))
    title = "Learning Curves for " + classifier_name + " Classifiers\n" + subtitle
    plt.title(title)
    plt.legend()

    filename = "LC_" + classifier_name + "_" + metric_name + "_metric.png"
    filename = filename.replace(" ", "_")
    filepath = os.path.join("results", filename)
    plt.savefig(filepath)
    plt.show()


def plot_roc_curve(classifier_name, fpr, tpr, auc, label = None, color = None):
    """
    Plots Receiver Operating Characteristic curve (ROC curve)
    :param classifier_name: String for the name of the classifier that will appear on the plot
    :param fpr: False Positive Rate values (x-axis)
    :param tpr: True Positive Rate values  (y axis
    :param auc: Area Under Curve value
    :param label: Generally not used
    :param color: Colour for the curve
    :return: None
    """

    label = classifier_name + " AUC: " + str(np.around(auc,3))
    title = "Receiver Operating Characteristic (ROC) Curve \nfor " + classifier_name

    plt.plot(fpr, tpr, linewidth=2, color =color, label= label)
    plt.plot([0,1], [0,1], color = 'silver',  linestyle= '--', label="Random Choice AUC: 0.5")
    plt.axis([0,1,0,1])
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.title(title)

    # Determine file name and path
    filename = "ROC_" + classifier_name + ".png"
    filename = filename.replace(" ", "_")
    filepath = os.path.join("results", filename)

    # Save to file
    plt.savefig(filepath)
    plt.show()