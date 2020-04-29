#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:37:54 2020

@author: antonio
"""

from sklearn.feature_extraction.text import CountVectorizer
import kmeans_elbow
from utils import argparser, docs2list, print_clusters, save_cluster_info

if __name__ == '__main__':
    
    ##### Parse arguments  
    datapath, mode, k = argparser()
    #datapath = '/home/antonio/Documents/Work/BSC/Projects/COVID-19/annotation-radiologia/data/bunch2'
    
    ##### Documents to list
    # TODO: further clean plain text
    all_documents, filenames = docs2list(datapath)
    
    ##### Text 2 Vector (Bag of Words)
    # TODO: substitute BOW by something more sofisticated (Doc2Vec, TF-iDC?)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_documents)

    ##### KMeans
    if mode == 'findK':
        kmeans_elbow.elbow(X.toarray())
        
    elif mode == 'clustering':
        kmeans, X_reduced = kmeans_elbow.clustering(X.toarray(), k)
        
        # Plot 
        # (from https://github.com/jordiae/DocumentClustering/blob/master/src/tag_corpus.py)
        print_clusters(kmeans, X_reduced, 'clustered.png')
        
        # Save clusters information
        save_cluster_info('clusters.txt', kmeans, filenames)