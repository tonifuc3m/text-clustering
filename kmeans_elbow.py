#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:27:22 2020

@author: antonio

KMeans with elbow method to find best K
https://pythonprogramminglanguage.com/kmeans-elbow-method/
"""


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import umap

def elbow(X):
       
    # Normalize data 
    # (for normalized vectors cosine similarity and euclidean similarity are connected linearly)
    X_scaled = preprocessing.scale(X)
    
    # k means determine k
    distortions = []
    K = range(1,16)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=0).fit(X_scaled)

        distortions.append(sum(np.min(cdist(X_scaled, kmeanModel.cluster_centers_,
                                            'cosine'), axis=1)) / X.shape[0])
    
    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    #plt.show()
    plt.savefig('elbow.png')
    
def clustering(X, k):
    
    # Normalize data 
    # (for normalized vectors cosine similarity and euclidean similarity are connected linearly)
    X_scaled = preprocessing.scale(X)
    
    # Reduce dimensionality (to be able to plot)
    reducer = umap.UMAP(metric='cosine', random_state=0)
    X_reduced = reducer.fit_transform(X_scaled)
    
    # Kmeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_reduced)
    
    return kmeans, X_reduced
    
    