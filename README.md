## Classifying Pulsars versus Non-Pulsars with Supervised Machine Learning


Pulsars are a rare type of neutron star that emit radio waves.  The HRTU2 data set was collected during the High Time Resolution Universe Survey (South) and contains samples of pulsar candidates.  The data set is unbalanced, containing 16, 259 spurious examples caused by RFI/noise and 1,639 real pulsar examples, giving a baseline accuracy of 90.8%.

The data set was used to make 5 predictive classifiers using the Scikit-learn toolkit.  
1. Decision tree
2. Random forest
3. Support Vector Machine
4. AdaBoost ensemble of decision trees
5. Naive Bayes.

The best hyperparameters were found with a cross validated grid search and then learning curves were plotted to identify overfitting and the model simplified if necesary. 

### Results

:-------------------------:|:-------------------------:
Decision tree learning curves             | Decision tree ROC curve
![Decision Tree Learning Curve](results/LC_Decision_Tree_error_rate_metric.png "Decision Tree Learning Curve") | ![Decision Tree ROC](results/ROC_Decision_Tree_Classifier.png "Decision Tree ROC")
Random forest learning curves             | Random forest ROC curve
![Random Forest Learning Curve](results/LC_Random_Forest_error_rate_metric.png "Random Forest Learning Curve") | ![Random Forest ROC](results/ROC_Random_Forest_Classifier.png "Random Forest ROC")
SVM learning curves             | SVM ROC curve
![Support Vector Learning Curve](results/LC_Support_Vector_Machine_error_rate_metric.png "SVM Learning Curve") | ![Support Vector Machine ROC](results/ROC_Support_Vector_Machine_Classifier.png "SVM ROC")
AdaBoost Learning Curves   | AdaBoost ROC curve
![AdaBoost Learning Curve](results/LC_Ada_Boost_error_rate_metric.png "AdaBoost Learning Curve") | ![AdaBoost ROC](results/ROC_Ada_Boost_Classifier.png "AdaBoost ROC")


#### Metrics

Metric        |   Accuracy   |  Error rate  |  Precision   |  Recall      |  F1          |AUC
:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:
Decision tree |0.980         |0.020         |0.964         |0.8170        |0.885         |0.973