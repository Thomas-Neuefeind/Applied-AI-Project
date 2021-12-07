from re import M
import numpy as np
from numpy.core.numeric import cross
import pandas as pd
import pylab as pl
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def fine_grid():
    """
    There is no input for this function
    This function is for the manipulation of fine grid tuning parameters

    @return a fine grid tuning parameter list 
    """
    tuned_parameters = [
        {
            'n_clusters' : [2],
            'random_state' : [22]
        }
        ]
    
    return(tuned_parameters)

def coarse_grid():
    """
    There is no input for this function
    This function is for the manipulation of coarse grid tuning parameters

    @return a coarse grid tuning parameter list 
    """
    tuned_parameters = [
        {
            'n_clusters' : [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            'random_state' : [22]
        }
        ]
    
    return(tuned_parameters)

def coarse_grid_results(X_train, Y_train, X_test, Y_test):
    """
    Input is the X, Y train and test datasets
    This function runs through the coarse_grid parameters on
    the X_train and Y_train datasets. Then will print out the results
    of which combination of parameters create the best fitting model
    

    @param X_train: The feature space of the training data
    @param Y_train: The classification space of the training data
    @param X_test: The feature space of the testing data
    @param Y_test: the classification space of the testing data
    @returns nothing, but prints out the best and all results from fitting
    the SVM model from the grid_params specificed by the coarse_grid function
    """
    
    
    # inputing the coarse grid parameters and evaluation criteria
    grid_params = coarse_grid()
    scores = ['accuracy', 'precision', 'recall']
    # creating the clf to go through
    clf = GridSearchCV(estimator=KMeans(), param_grid=grid_params, scoring='accuracy', cv=5)
    # fitting to the clf
    clf.fit(X_train, Y_train)
    
    # printing out which does the best using the specified metrics
    print("Best parameters set found on development set (via coarse_grid): \n")
    print(clf.best_params_)
    print()
    print("Grid scores on development set: \n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()


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
    
    
    # shows the results from the coarse grid
    #coarse_grid_results(x_train, y_train, x_test, y_test)
    
    
    
    clf = KMeans(n_clusters=2, random_state=22)
    
    # fitting the knn Classifier
    
    clf.fit(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    
    #plot_confusion_matrix(y_test, y_pred)
    
    print('Test Accuracy : %.3f'%clf.score(x_test, y_test)) ## Score method also evaluates accuracy 

if __name__ == '__main__':
    main()