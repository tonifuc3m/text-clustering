#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:34:33 2020

@author: antonio
utils
"""

from spacy.lang.es import Spanish
from spacy.lang.es import STOP_WORDS
from string import punctuation
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import argparse
import os


def Flatten(ul):
    '''
    DESCRIPTION: receives a nested list and returns it flattened
    
    Parameters
    ----------
    ul: list
    
    Returns
    -------
    fl: list
    '''
    
    fl = []
    for i in ul:
        if type(i) is list:
            fl += Flatten(i)
        else:
            fl += [i]
    return fl



def tokenize(text):
    '''
    Tokenize a string in Spanish

    Parameters
    ----------
    text : str
        Spanish text string to tokenize.

    Returns
    -------
    tokenized : list
        List of tokens (includes punctuation tokens).

    '''
    nlp = Spanish()
    doc = nlp(text)
    token_list = []
    for token in doc:
        token_list.append(token.text)
    return token_list


def print_clusters(kmeans, X_reduced, filepath):
    '''
    Extracted from: 
    https://github.com/jordiae/DocumentClustering/blob/master/src/tag_corpus.py
    Prints Kmeans (from data in 2D)
    
    Parameters
    ----------
    kmeans : sklearn.cluster._kmeans.KMeans
        KMeans object, output of Kmeans fit method
    X_reduced: numpy.ndarray
        Numpy array with dimensions (n, 2), where n is the number of points to 
        print.
    filepath: string
        Path to output PNG file with clustering

    Returns
    -------
    None
    '''
    
    
    c = [sns.color_palette()[x] for x in kmeans.labels_]
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=c)
    def get_color_map(cc, labels, k=8):
        colormap = {}
        for label, ccc in zip(labels, cc):
            if label in colormap:
                continue
            else:
                colormap[label] = ccc
            if len(colormap) == k:
                break
        return colormap
    colormap = get_color_map(c,kmeans.labels_,8)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('2D unsupervised UMAP projection', fontsize=24)
    patches = []
    for label, color in colormap.items():
        patch = mpatches.Patch(color=color, label=label)
        patches.append(patch)
    plt.legend(handles=patches)
    #plt.show()
    plt.savefig(filepath)
    
def argparser():
    
    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.add_argument("-i", "--datapath", required = True, dest = "datapath", 
                        help = "absolute path to folder with plain text files")
    parser.add_argument('-m', '--mode', required=True, dest = 'mode',
                        choices=['findK','clustering'], 
                        help = "Run mode: either show plot to decide best K or do clustering")
    parser.add_argument("-k", required = False, default='5', dest = "k", 
                        help = "number of clusters")
    
    datapath = parser.parse_args().datapath
    mode = parser.parse_args().mode
    k = int(parser.parse_args().k)
    
    return datapath, mode, k


def docs2list(datapath):
    '''
    Concatenates all text files in datapath into a list. 
    One text file per entry.
    Remove stopwords and punctuation
    
    Parameters
    ----------
    datapath: string
        Path to directory where text files are

    Returns
    -------
    all_documents: list
        List with all text file content in a list. One entry per text  file
    filenames: list
        List with text file names
    '''
    
    all_documents = []
    filenames = []
    for root,d,files in os.walk(datapath):
        for file in files:
            if file[-3:] != 'txt':
                continue
            txt = open(os.path.join(root, file)).readlines()
            txt_clean = list(map(lambda x: x.strip(), txt))
            words = Flatten(list(map(lambda x: tokenize(x), txt_clean)))
            words_stw = list(filter(lambda x: x not in STOP_WORDS, words))
            words_punct = list(filter(lambda x: x not in punctuation, words_stw))
            all_documents.append(' '.join(words_punct))
            filenames.append(file)
    return all_documents, filenames


def save_cluster_info(filepath, kmeans, filenames):
    '''
    Create text file with: 
        filename #cluster
    
    Parameters
    ----------
    filepath: string
        Path to output text file
    kmeans: sklearn.cluster._kmeans.KMeans
        KMeans object, output of Kmeans fit method
    filenames: list
        List with text file names
    
    Returns
    -------
    None
    '''
    with open(filepath, 'w') as f:
        s = ''
        for index, label in enumerate(kmeans.labels_):
            s += filenames[index] + ' ' + str(label) + '\n'
        f.write(s)
