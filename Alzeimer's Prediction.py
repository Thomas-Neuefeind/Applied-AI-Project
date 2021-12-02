from re import M
import numpy as np
from numpy.core.numeric import cross
import pandas as pd
import pylab as pl
from sklearn import svm, datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from SVM_Prediction import SVM_Predict

cross_sectional = pd.read_csv("oasis_cross-sectional.csv")
longitudinal = pd.read_csv("oasis_longitudinal.csv")


#filter any rows with missing data
cross_sectional.dropna(axis = 0, how = 'any', inplace=True)
longitudinal.dropna(axis = 0, how = 'any', inplace=True)

cross_sectional= cross_sectional.drop(columns=['ID', 'Hand', 'Delay'])
longitudinal = longitudinal.drop(columns=['Subject ID', 'MRI ID', 'MR Delay', 'Visit', 'Hand'])
longitudinal = longitudinal.rename(columns={'EDUC':'Educ'})

data = pd.concat([cross_sectional, longitudinal])

svm_predicter = SVM_Predict()
svm_predicter.predict(data)

# labelencoder = LabelEncoder()
# for row in data :
#     data[row] = labelencoder.fit_transform(data[row])

# features = ['M/F', 'Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
# X = data[features]
# Y = data['CDR']

# # Use Sklearn to get splits in our data for training and testing. --this is setting up the k-fold process!
# # test_size of 0.8 means 80% of data remains training data with the other 20% as test data
# x_train, x_test, y_train, y_test = train_test_split (X, Y, test_size = 0.8, random_state=22)
# y_train_converted = y_train.values.ravel()

# tuned_parameters = [
#     {
#         'kernel': ['linear'], 
# #        'C': [1, 10, 25, 50, 100, 1000]
#         'C': [10, 25, 50, 100]
#     },
#     # {
#     #    'kernel': ['poly'], 
#     #    'degree': [1,2,3, 4],
#     #    'C': [1, 10, 100, 1000]
#     # },
#     # {
#     #     'kernel': ['rbf'], 
#     #     'gamma': [1e-3, 1e-4],
#     #     'C': [1, 10, 100, 1000]
#     # }
# ]

# scores = ['precision', 'recall']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#     clf = GridSearchCV(
#         SVC(), tuned_parameters, scoring='%s_macro' % score
#     )
#     clf.fit(x_train, y_train)

#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()

#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(x_test)
#     print(classification_report(y_true, y_pred))
#     print()