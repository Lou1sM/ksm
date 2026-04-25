from kstar_means import KStarMeans
import numpy as np


X = np.random.rand(10000, 2) # the method works best for low-dim data, strongly recommended to use UMAP to reduce dimension to 2

ksm = KStarMeans()
cluster_labels = ksm.fit_predict(X)
# as there is no structure in this random data, it is expected to find just one cluster
