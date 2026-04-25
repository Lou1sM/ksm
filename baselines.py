from tqdm import tqdm
from time import time
import math
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import densitypeakclustering as dc
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.spatial import KDTree
from dl_utils.label_funcs import compress_labels

from kstar_means import compute_cluster_cost

LOG2PI = 1.837877

class CRP:
   def __init__(self, alpha=1.0, n_iter=3):
       self.alpha = alpha
       self.n_iter = n_iter

   def fit_predict(self, X):
       n, d = X.shape

       # Better initialization: start with fewer clusters
       n_initial_clusters = min(10, n // 10, n)
       clusters = np.random.randint(0, n_initial_clusters, n)

       for _ in range(self.n_iter):
           #for i in tqdm(range(n)):
           for i in range(n):
               # Vectorized cluster counting (excluding point i)
               mask = np.arange(n) != i
               other_clusters = clusters[mask]
               unique_clusters, counts = np.unique(other_clusters, return_counts=True)
               cluster_counts = dict(zip(unique_clusters, counts))

               # Calculate probabilities
               probs = {}

               # Existing clusters - vectorized similarity computation
               for cluster_id, count in cluster_counts.items():
                   cluster_mask = (clusters == cluster_id) & mask
                   if np.any(cluster_mask):
                       cluster_center = X[cluster_mask].mean(axis=0)
                       similarity = cosine_similarity([X[i]], [cluster_center])[0, 0]
                       probs[cluster_id] = count * max(similarity, 0.01)

               # New cluster probability
               new_cluster_id = clusters.max() + 1
               probs[new_cluster_id] = self.alpha

               # Normalize and sample
               total_prob = sum(probs.values())
               probs = {k: v/total_prob for k, v in probs.items()}

               # Sample new cluster
               rand_val = random.random()
               cumsum = 0
               for cluster_id, prob in probs.items():
                   cumsum += prob
                   if rand_val <= cumsum:
                       clusters[i] = cluster_id
                       break

       # Relabel consecutively
       unique_clusters = np.unique(clusters)
       cluster_map = {old: new for new, old in enumerate(unique_clusters)}
       return np.array([cluster_map[c] for c in clusters])


class DensityPeaksClustering:
    def __init__(self):
        self

    def fit_predict(self, X_):
        np.NaN = np.nan
        if len(X_) > 70000:
            X = X_[np.random.choice(len(X_), 70000)]
        else:
            X = X_
        D = dc.distance_matrix(X)

        d_c = 0.2
        # Calculate the local density
        rho = dc.local_density(D, d_c)

        # Calculate the minimum distance to a point with higher density
        delta,nearest = dc.distance_to_larger_density(D, rho)

        rho_min = 2
        delta_min = 0.2
        centers = dc.cluster_centers(rho, delta, rho_min=rho_min, delta_min=delta_min)

        # Assign cluster ID's to all datapoints
        if len(X_) > 70000:
            centroids = X[centers]
            dists = ((np.expand_dims(X_, 1) - np.expand_dims(centroids, 0))**2).sum(axis=2)
            ids = dists.argmin(axis=1)
        else:
            ids = dc.assign_cluster_id(rho, nearest, centers)
            labels, _, _ = compress_labels(ids)
        return ids


class DivisiveHierarchical:
    def __init__(self, kmax=10, min_silhouette=0.5):
        self.kmax = kmax
        self.min_silhouette = min_silhouette

    def fit_predict(self, X):
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        current_clusters = 1
        best_labels = labels.copy()
        best_silhouette = -1

        while current_clusters < self.kmax:
            max_var, split_idx = -1, -1
            for i in range(current_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 1:
                    var = np.var(cluster_points, axis=0).sum()
                    if var > max_var:
                        max_var, split_idx = var, i

            if split_idx == -1:
                break

            cluster_points = X[labels == split_idx]
            cluster_indices = np.where(labels == split_idx)[0]
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=2, random_state=0).fit(cluster_points)
            new_labels = kmeans.labels_

            temp_labels = labels.copy()
            new_cluster_id = current_clusters
            temp_labels[cluster_indices] = np.where(new_labels == 0, split_idx, new_cluster_id)
            silhouette = silhouette_score(X, temp_labels) if current_clusters + 1 <= n_samples else -1

            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_labels = temp_labels.copy()

            if silhouette < self.min_silhouette:
                break

            labels[cluster_indices] = np.where(new_labels == 0, split_idx, new_cluster_id)
            current_clusters += 1

        return best_labels

class Pymc3DPMM():
    def fit_predict(self, X):
        import pymc3 as pm
        data = np.random.randn(10000, 2)
        with pm.Model():
            alpha = 1.0
            mu = pm.Normal('mu', 0, sd=10, shape=(10, 2))
            sd = pm.HalfNormal('sd', sd=1, shape=10)
            weights = pm.Dirichlet('weights', a=np.ones(10))
            cat = pm.Categorical('cat', p=weights, shape=10000)
            obs = pm.MvNormal('obs', mu=mu[cat], cov=np.eye(2)*sd[cat], observed=data)
            trace = pm.sample(1000, tune=1000)

        clusters = trace['cat'].mean(axis=0).round().astype(int)
        return clusters

def compute_two_part_cost(X, k, mahala_dists_ish):
    n, nz = X.shape
    cluster_cost = compute_cluster_cost(X)
    resid_cost = 0.5 * (nz*LOG2PI + mahala_dists_ish)
    idx_cost = n*np.log(k)
    cost = k*cluster_cost + idx_cost + resid_cost
    return cost

class KMeansMDLSweep():
    def fit_predict(self, X):
        N, nz = X.shape
        best_cost = np.inf
        best_k = -1
        for k in tqdm(range(1,math.ceil(N**0.5)+1)):
            kmeans_model = KMeans(k)
            kmeans_model.fit(X)
            mahala_dists_ish = kmeans_model.inertia_
            cost = compute_two_part_cost(X, k, mahala_dists_ish)
            if cost < best_cost:
                best_k = k
                best_cost = cost

        clusters = KMeans(best_k).fit_predict(X)
        return clusters


class QuickShift():
    def __init__(self, bandwidths=np.arange(0.1, 2.0, 0.5), taus=np.arange(1.5, 3.0, 0.5)):
        self.bandwidths = bandwidths
        self.taus = taus

    def _density(self, X, bandwidth):
        nn = NearestNeighbors(radius=bandwidth).fit(X)
        counts = nn.radius_neighbors(X, return_distance=False)
        return np.array([len(c) for c in counts])  # neighbour count ≈ density

    def _cluster(self, X, bandwidth, tau, density):
        tree = KDTree(X)
        n = len(X)
        parent = np.arange(n)

        idxs_list = tree.query_ball_point(X, r=tau)

        for i in tqdm(range(n)):
            neighbours = [j for j in idxs_list[i] if density[j] > density[i]]
            if not neighbours:
                continue
            dists = np.linalg.norm(X[neighbours] - X[i], axis=1)
            parent[i] = neighbours[np.argmin(dists)]

        labels = np.arange(n)
        for i in range(n):
            root = i
            while parent[root] != root:
                root = parent[root]
            labels[i] = root

        _, labels = np.unique(labels, return_inverse=True)
        return labels

    def fit_predict(self, X):
        best_labels, best_bic = None, np.inf

        for bw in self.bandwidths:
            print('bandwidth', bw)
            #kde = KernelDensity(bandwidth=bw).fit(X)
            #density = np.exp(kde.score_samples(X))
            density = self._density(X, bw)
            for tau in self.taus:
                print('tau', tau)
                labels = self._cluster(X, bw, tau, density)
                n_clusters = len(np.unique(labels))

                if n_clusters < 2 or n_clusters >= len(X):
                    continue

                bic = GaussianMixture(n_components=n_clusters, max_iter=20).fit(X).bic(X)
                if bic < best_bic:
                    best_bic, best_labels, self.best_bw, self.best_tau = bic, labels, bw, tau

        return best_labels
