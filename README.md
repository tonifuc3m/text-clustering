Install requirements: 

```
pip install -r requirements.txt
```


### Steps: 
+ Read text files
+ Bag of Words
+ Normalize Bag of Words (so that euclidean distance is equivalent to cosine distance)
+ UMAP projection to 2D (to be able to visualize clusters)
+ KMeans


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

### TODOs:
+ Enhance BOW: very memory consuming...highly sparse matrices
+ Substitute BOW by Doc2Vec with initialized word embeddings
+ Do KMeans on the original 200D, not on the projected 2D space. Once I have applied KMeans, I can project to 2D and visualize it with colors.
