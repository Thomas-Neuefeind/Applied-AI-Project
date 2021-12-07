from re import M
import numpy as np
from numpy.core.numeric import cross
import pandas as pd
import pylab as pl
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


def plot_confusion_matrix(Y_test, Y_preds):
    conf_mat = confusion_matrix(Y_test, Y_preds)
    #print(conf_mat)
    fig = plt.figure(figsize=(6,6))
    plt.matshow(conf_mat, cmap=plt.cm.Blues, fignum=1)
    plt.yticks(range(4), range(4))
    plt.xticks(range(4), range(4))
    plt.colorbar();
    for i in range(4):
        for j in range(4):
            plt.text(i-0.2,j+0.1, str(conf_mat[j, i]), color='tab:red')




def main():
    cross_sectional = pd.read_csv("oasis_cross-sectional.csv")
    longitudinal = pd.read_csv("oasis_longitudinal.csv")


    #filter any rows with missing data
    cross_sectional.dropna(axis = 0, how = 'any', inplace=True)
    longitudinal.dropna(axis = 0, how = 'any', inplace=True)

    cross_sectional= cross_sectional.drop(columns=['ID', 'Hand', 'Delay'])
    longitudinal = longitudinal.drop(columns=['Subject ID', 'MRI ID', 'MR Delay', 'Visit', 'Hand' , 'Group'])
    longitudinal = longitudinal.rename(columns={'EDUC':'Educ'})

    data = pd.concat([cross_sectional, longitudinal])
    
    
    labelencoder = LabelEncoder()
    for row in data :
        data[row] = labelencoder.fit_transform(data[row])

    features = ['M/F', 'Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
    X = data[features]
    Y = data['CDR']
        # Use Sklearn to get splits in our data for training and testing. --this is setting up the k-fold process!
        # test_size of 0.8 means 80% of data remains training data with the other 20% as test data
    x_train, x_test, y_train, y_test = train_test_split (X, Y, test_size = 0.8, random_state=22)
    y_train_converted = y_train.values.ravel()
    
    clf = GaussianNB()
    
    # fitting the gaussian naive bayes
    clf.fit(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    
    #plot_confusion_matrix(y_test, y_pred)
    
    print('Test Accuracy : %.3f'%clf.score(x_test, y_test)) ## Score method also evaluates accuracy 
    
if __name__ == '__main__':
    main()

