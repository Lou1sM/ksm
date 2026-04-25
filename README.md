This is the code for the method from the paper [K∗
-Means: A Parameter-free Clustering Algorithm](https://arxiv.org/pdf/2505.11904), a variant of $k$-means that doesn't require specifying $k$. 

While some existing clustering algorithms can automatically determine the number of clusters, they require setting other, unintuitive parameters. This just kicks the problem down the road, because these other parameters end up determining the number of clusters found. As we show in the paper, for example, DBSCAN can produce anywhere from 6 to several thousand clusters on MNIST, depending on the min_pts and eps parameters. 

$K^*$-means is, to our knowledge, the first entirely parameter-free clustering algorithm.

The code for the model is in `kstar_means.py`. You can run it simply as
```
from kstar_means import KStarMeans

ksm = KStarMeans()
ksm.fit_predict(<your-data>)
```

Each data point should be flattened, so the tensor passed to fit predict is of shape (N, nz), where N is the number of data points and nz the dimensionality. It has only been tested and shown to work when nz=2. It is recommended to use UMAP for dimensionality reduction.
