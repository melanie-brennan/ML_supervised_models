import util
import model_dt
import model_nb
import model_rf
import model_svm
import model_boost


#download the High Time Resolution Universe Survey (South) (HTRU2) from UCI Machine Learning Repository
zip_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip"
util.download_extract_zip(zip_url)

#create a training and test set from the csv file
(X_train, X_test, y_train, y_test) = util.create_train_test_sets("HTRU_2.csv", test_proportion = 0.2, label_column = -1)

# Use Models
# 1. Decision Tree
model_dt.use_dt(X_train, y_train, X_test, y_test)

# 2. Random Forest
model_rf.use_rf(X_train, y_train, X_test, y_test)

# 3. Support Vector Machine
model_svm.use_svm(X_train, y_train, X_test, y_test)

# 4. Ada Boost
model_boost.use_boost(X_train, y_train, X_test, y_test)

# 5. Gaussian Naive Bayes
model_nb.perform_nb(X_train, y_train, X_test, y_test)