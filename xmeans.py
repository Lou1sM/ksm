import numpy as np
from pyclustering.cluster.xmeans import xmeans as xmeans_lib
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer



class DummyXMeans():
    def __init__(self, kmax):
        self.kmax = kmax

    def fit_predict(self, X):
        initial_centers = kmeans_plusplus_initializer(X, 2).initialize()
        xmeans_instance = xmeans_lib(X, initial_centers, kmax=self.kmax)

        # Run clustering
        xmeans_instance.process()

        # Get clusters and centers
        clusters = xmeans_instance.get_clusters()
        self.cluster_labels = np.zeros(len(X)).astype(int)
        for i, cluster in enumerate(clusters):
           self.cluster_labels[cluster] = i
        self.centers = xmeans_instance.get_centers()
        return self.cluster_labels

