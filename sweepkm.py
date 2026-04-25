from sklearn.cluster import KMeans
from time import time
from kneed import KneeLocator
import math
import numpy as np
from tqdm import tqdm


class KMeansBICSweep():
    def fit_predict(self, X):
        N = len(X)
        #pow_incr = 1.1
        #max_pow = math.ceil(np.log(N/10) / np.log(pow_incr))
        #ks_to_sweep = list(set([int(pow_incr**i) for i in range(max_pow)]))
        ks_to_sweep = range(1,math.ceil(N**0.5)+1)
        best_bic = np.inf
        for k in tqdm(ks_to_sweep):
            self.km = KMeans(n_clusters=k)
            self.km.fit(X)
            logL = self.km.inertia_
            bic = k*np.log(N) + logL # 0.5 factor and d=2 cancel each other out
            if bic < best_bic:
                best_bic = bic
                best_k = k

        self.best_km = KMeans(n_clusters=best_k)
        preds = self.best_km.fit_predict(X)
        return preds

class ElbowMethod():
    def fit_predict(self, X):
        #ks_to_sweep = range(1, int(len(X)**0.5))
        ks_to_sweep = range(1, 200, 2)
        kmeans_costs = []
        for k in tqdm(ks_to_sweep):
            self.km = KMeans(n_clusters=k)
            kmc = self.km.fit(X).inertia_
            kmeans_costs.append(kmc)
        kl = KneeLocator(ks_to_sweep, kmeans_costs, curve='convex', direction='decreasing')
        best_k = kl.elbow
        self.best_km = KMeans(n_clusters=best_k)
        preds = self.best_km.fit_predict(X)
        return preds
