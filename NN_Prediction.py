## Neural Net prediction -- most of this code is taken from Project 3 in COSC 523 with Dr. Williams
from re import M
import numpy as np
from numpy.core.numeric import cross
import pandas as pd
from pandas.core.arrays import string_
import pylab as pl
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


class NN_Predict :
    
    def __init__(self) -> None:
        pass
    
    def predict(self, data) :

        # (0) Hide as many warnings as possible! 
        import os
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('INFO')
        tf.compat.v1.disable_eager_execution()

       # cleanData = pd.DataFrame(data)

        # cleanData.replace({'M/F' : 'M'}, 1)
        # cleanData.replace({'M/F' : 'F'}, 0)
        #cleanData.replace({'CDR' : 0}, 1)
        #cleanData.replace({'CDR' : 0.5}, 2)
        #cleanData.replace({'CDR' : 1}, 3)
        #cleanData.replace({'CDR' : 1.5}, 4)
#        cleanData.replace({'CDR' : 2}, 5)

        #cleanData= cleanData.drop(columns=['CDR'])
        # for row in data : 
        #     for col in row :
        #         if(isinstance(col,str)):
        #             col = float(col)
        #         elif(isinstance(col, float)):
        #             col = int(col)
        #         else :
        #             pass

        labelencoder = LabelBinarizer()
        for row in data :
            data[row] = labelencoder.fit_transform(data[row])

        features = ['M/F', 'Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
        X = data[features]
        Y = data['CDR']

        # Use Sklearn to get splits in our data for training and testing. --this is setting up the k-fold process!
        # test_size of 0.8 means 80% of data remains training data with the other 20% as test data
        X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size = 0.8, random_state=22)
        y_train_converted = y_train.values.ravel()

        # Perform standardization on our data.
        scaler = MinMaxScaler(feature_range=(0,1))
        X_train = pd.DataFrame(scaler.fit_transform(X_train),
                                    columns=X_train.columns,
                                    index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test),
                                columns=X_test.columns,
                                index=X_test.index)

        def DynamicModel(neuron_one=1, neuron_two=1, activation_one='sigmoid', activation_two='sigmoid'):
            """ A sequential Keras model that has an input layer, one 
                hidden layer with a dymanic number of units, and an output layer."""
            model = Sequential()
            model.add(Dense(neuron_one, input_dim=4, activation=activation_one, name='layer_1'))
            model.add(Dense(neuron_two, activation=activation_two, name='layer_2'))
            model.add(Dense(neuron_two, activation=activation_two, name='layer_2'))
            model.add(Dense(2, activation='sigmoid', name='output_layer'))
            
            # Don't change this!
            model.compile(loss="categorical_crossentropy",
                        optimizer="adam",
                        metrics=['accuracy'])
            return model

        # (6) Evaluation + HyperParameter Search
        # Below, we build KerasClassifiers using our model definitions. Use verbose=2 to see
        # real-time updates for each epoch.
        model = KerasClassifier(
            build_fn=DynamicModel, 
            epochs=200, 
            batch_size=20, 
            verbose=0)

        # (7) Define a set of unit numbers (i.e. "neurons") and activation functions
        # that we want to explore.
        param_grid = [
            {
                'activation_one': ['relu', 'tanh', 'linear'], 
                'activation_two': ['relu', 'tanh', 'linear'], 
                'neuron_one': [10],
                'neuron_two': [15, 20]
            }
        ]

        # (8)   Send the Keras model through GridSearchCV, and evaluate the performnce of every option in 
        #       param_grid for the "neuron" value.
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, verbose=2)
        grid_result = grid.fit(X_train, y_train)

        # (9) Print out a summarization of the results.
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))