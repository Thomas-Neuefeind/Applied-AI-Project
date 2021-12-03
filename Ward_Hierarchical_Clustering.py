### Ward Hierarchical Clustering -- Alzheimer's prediction
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class WHC_Prediction : 
    def __init__(self) :
        pass

    def predict(self, data) :
        
        data = data.iloc[:, [3,4]].values

        wardClustering = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
        y_wardClustering = wardClustering.fit_predict(data)

        print(y_wardClustering)
        
        # plt.scatter(data[y_wardClustering==0,0], data[y_wardClustering==0,1], s=100, label= "cluster 1")
        # plt.scatter(data[y_wardClustering==1,0], data[y_wardClustering==1,1], s=100, label= "cluster 2")
        # plt.scatter(data[y_wardClustering==2,0], data[y_wardClustering==2,1], s=100, label= "cluster 3")
        # plt.scatter(data[y_wardClustering==3,0], data[y_wardClustering==3,1], s=100, label= "cluster 4")
        # plt.scatter(data[y_wardClustering==4,0], data[y_wardClustering==4,1], s=100, label= "cluster 5")
        # plt.scatter(data[y_wardClustering==5,0], data[y_wardClustering==5,1], s=100, label= "cluster 6")
        # plt.scatter(data[y_wardClustering==6,0], data[y_wardClustering==6,1], s=100, label= "cluster 7")
        # plt.show()
