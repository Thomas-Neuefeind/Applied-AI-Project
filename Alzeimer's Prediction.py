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
from Ward_Hierarchical_Clustering import WHC_Prediction


cross_sectional = pd.read_csv("oasis_cross-sectional.csv")
longitudinal = pd.read_csv("oasis_longitudinal.csv")


#filter any rows with missing data
cross_sectional.dropna(axis = 0, how = 'any', inplace=True)
longitudinal.dropna(axis = 0, how = 'any', inplace=True)

cross_sectional= cross_sectional.drop(columns=['ID', 'Hand', 'Delay'])
longitudinal = longitudinal.drop(columns=['Subject ID', 'MRI ID', 'MR Delay', 'Visit', 'Hand'])
longitudinal = longitudinal.rename(columns={'EDUC':'Educ'})

data = pd.concat([cross_sectional, longitudinal])

#svm_predicter = SVM_Predict()
#svm_predicter.predict(data)

ward_predicter = WHC_Prediction()
ward_predicter.predict(data)