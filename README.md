Warning: requirements.txt has more stuff than needed. List of python packages:
numpy, spacy, umap, sklearn, sentence-splitter, matplotlib, seaborn, scipy

### Steps: 
Read text files
Bag of Words
Normalize Bag of Words
KMeans


### Usage
Two modes: 
+ findK
Compute KMeans with different K and plot distortions to decide by elbow method the best K
```
python main.py -i /path/to/directory/ -m findK
```

+ clustering
If I already know my number of clusters, compute KMeans
```
python main.py -i /path/to/directory/ -m clustering -k K
```

