import os
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from dl_utils.label_funcs import accuracy
from time import time
import numpy as np
import torch
from utils import profile_lines, print_stats
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

LOG2PI = 1.837877
#torch.set_default_dtype(torch.float64)
torch.set_float32_matmul_precision('high')

# mahala_dists_ish means the sum of all unnormed mahalanobis distances, i.e.

def compute_cluster_cost(X):
    nz = X.shape[1]
    uniq_x = np.unique(X.flatten()).astype(float)
    precision_to_use = min(np.sort(uniq_x)[1:] - np.sort(uniq_x)[:-1])
    prec_cost = -np.log(precision_to_use)
    #self.prec_cost = 32
    range_size = X.max() - X.min()
    float_cost = prec_cost + np.log(range_size)
    cluster_cost = nz * float_cost
    return cluster_cost

class KStarMeans():
    def __init__(self, subinit, n_init=1, nonfixed_idx_cost=False, nonfixed_sig=False, dist_diffs=False, compute_full_densities=False, compute_outliers=False, verbose=False, run_checks=False):
        self.device = torch.device('cpu')
        self.n_init = n_init
        self.nonfixed_idx_cost = nonfixed_idx_cost
        self.nonfixed_sig = nonfixed_sig
        self.compute_full_densities = compute_full_densities
        self.run_checks = run_checks
        self.compute_outliers = compute_outliers
        self.subinit = subinit
        self.dist_diffs = dist_diffs
        self.verbose = verbose

    @property
    def nc(self):
        return self.clusters.shape[0]

    def compute_resid_cost(self, sq_dists, sig=None):
        if self.nonfixed_sig:
            assert sig is not None
            log_cov_det = sig.log().sum()
            mahala_dists_ish = (sq_dists/sig).sum(axis=1)
        else:
            assert sig is None
            mahala_dists_ish = sq_dists.sum(axis=1)
            log_cov_det = 0
        resid_cost = 0.5 * (self.nz*LOG2PI + log_cov_det + mahala_dists_ish)
        #resid_cost = mahala_dists_ish / 2
        return resid_cost

    def mdl_cost(self):
        self.model_cost = self.cluster_cost*self.nc
        if self.nonfixed_idx_cost:
            self.cost_by_symbol = self.logn - self.clusters.sum(axis=1).log()
            self.idx_cost_by_c = self.cost_by_symbol * self.clusters.sum(axis=1)
            self.idx_cost = self.idx_cost_by_c.sum()
        else:
            self.idx_cost = self.n * np.log(self.nc)
        self.all_sq_cluster_dists = torch.cdist(self.X, self.centroids, p=2)**2
        resid_costs = 0.5 * (self.nz*LOG2PI + self.all_sq_cluster_dists)
        self.resid_cost_by_c = (resid_costs.T * self.clusters).sum(axis=1)
        if self.resid_cost_by_c.sum().isnan():
            breakpoint()
        return self.model_cost + self.idx_cost + self.resid_cost_by_c.sum()

    def new_subcentroid(self, i):
        assert i <= self.nc
        cluster_vecs = self.X[self.clusters[i].bool().T]
        if self.subinit=='kplusm':
            #best_new = cluster_vecs[torch.randint(0, len(cluster_vecs), (2,))]
            best_new1 = cluster_vecs[torch.randint(0, len(cluster_vecs), (1,))]
            probs = ((cluster_vecs - best_new1)**2).sum(axis=1)
            best_new2 = cluster_vecs[torch.multinomial(probs, 1)]
            best_new = torch.cat([best_new1, best_new2])
        elif self.subinit=='rand':
            best_new = self.centroids[i] + torch.randn(size=(2,1))
        else:
            assert self.subinit=='ksm'
            cluster_point_densities = self.point_densities[self.clusters[i].bool().T]
            for attempt in range(1):
                indices = torch.multinomial(cluster_point_densities, 2)
                new = (self.centroids[i] + cluster_vecs[indices]) / 2
                dist_from_subcentroids = ((cluster_vecs[:,None] - new)**2).sum(axis=2)
                summ_dist, assigned = dist_from_subcentroids.min(axis=1)
                if attempt==0 or (summ_dist.sum() < best_dist and len(assigned.unique())==2):
                    best_new = new
                    best_dist = summ_dist.sum()
            if len(assigned.unique())!=2:
                breakpoint()
        if self.nonfixed_sig:
            nv1 = cluster_vecs[assigned.bool()].var(axis=0)
            nv2 = cluster_vecs[~assigned.bool()].var(axis=0)
            new_var = torch.stack([nv1, nv2])
        else:
            new_var = torch.ones(2, self.nz, device=self.device)
        assert best_new.shape == (2, self.nz)
        return best_new, new_var

    def init_new_subcentroids(self, i):
        assert i < self.nc
        if self.clusters[i].sum()==1:
            self.subcentroids[i], self.subvariances[i] = torch.zeros_like(self.subcentroids[0]), torch.zeros_like(self.subvariances[0])
            return
        for attempt in range(100):
            self.subcentroids[i], self.subvariances[i] = self.new_subcentroid(i)
            self.assign_subclusters(i)
            if (self.subclusters[i].sum(axis=1)!=0).all():
                break
        self.update_subclusters(i)
        #if attempt>0:
            #print(f'init new subcs took {attempt+1} attempts')

    def update_subclusters(self, i):
        for j in range(2):
            cluster_vecs = self.X[self.subclusters[i,j].bool()]
            self.subcentroids[i,j] = cluster_vecs.mean(axis=0)
            self.subvariances[i,j] = cluster_vecs.var(axis=0) if self.nonfixed_sig else torch.ones(self.nz, device=self.device)

    #@profile_lines
    def assign_subclusters(self, i):
        #idx_long = torch.nonzero(self.clusters[i]).squeeze(1)
        idx_long = self.clusters[i].bool()
        if self.nonfixed_sig:
            dists = (((self.X[idx_long,None] - self.subcentroids[i])**2 / self.subvariances[i])**2).sum(axis=2) + self.subvariances[i].log().sum(axis=1)
        else:
            dists = torch.cdist(self.X[idx_long], self.subcentroids[i])
        subcluster_assignments = dists.argmin(axis=1)
        self.subclusters[i, :, :] = 0
        self.subclusters[i,subcluster_assignments,idx_long] = 1
        if self.run_checks:
            assert self.subclusters[i].sum() == self.clusters[i].sum()
        return dists

    #@profile_lines
    def kmeans_step(self):
        if self.run_checks:
            before = self.mdl_cost()
            before_mod = self.model_cost
            before_idx = self.idx_cost.item()
            before_resid = self.resid_cost_by_c.sum().item()
            before_clusters = self.clusters.clone()
            oa = before_clusters.argmax(axis=0)
            before_subclusters = self.subclusters.clone()
        if self.nonfixed_sig:
            mahala_dists_ish = ((self.X[:,None] - self.centroids)**2 / self.variances).sum(axis=2) + self.variances.log().sum(axis=1)
        else:
            mahala_dists_ish = torch.cdist(self.X, self.centroids)**2
        self.clusters = (torch.arange(self.nc, device=self.device).tile(self.n,1) == mahala_dists_ish.argmin(axis=1, keepdims=True)).long().T
        if self.run_checks and (from_assignment:=(self.mdl_cost() > before+1e-10)):
            print(f'mod: {before_mod:.3f}->{self.model_cost:.3f}, idx: {before_idx:.3f}->{self.idx_cost.item():.3f}, resid: {before_resid:.3f}->{self.resid_cost_by_c.sum().item():.3f}')
            if self.resid_cost_by_c.sum() > before_resid:
                ras = (self.clusters!=before_clusters).any(axis=0).nonzero().squeeze(1)
                if self.nonfixed_sig:
                    x=torch.stack([self.compute_resid_cost((self.X-self.centroids[i])**2, self.variances[i]) for i in range(self.nc)], axis=1)
                else:
                    x=torch.stack([self.compute_resid_cost((self.X-self.centroids[i])**2) for i in range(self.nc)], axis=1)
                na = self.clusters.argmax(axis=0)
                breakpoint()
        nonzero_mask = self.clusters.sum(axis=1)!=0
        if not nonzero_mask.all():
            self.clusters = self.clusters[nonzero_mask]; self.subclusters = self.subclusters[nonzero_mask] # some clusters may have become empty after reassignment
        self.centroids = torch.stack([self.X[self.clusters[i].bool()].mean(axis=0) for i in range(self.nc)])
        if self.nonfixed_sig:
            self.variances = torch.stack([self.X[self.clusters[i].bool()].var(axis=0) for i in range(self.nc)])
        if self.run_checks and self.mdl_cost() > before+1e-10 and not from_assignment:
            print(f'mod: {before_mod:.3f}->{self.model_cost:.3f}, idx: {before_idx:.3f}->{self.idx_cost.item():.3f}, resid: {before_resid:.3f}->{self.resid_cost_by_c.sum().item():.3f}')
            ras = (self.clusters!=before_clusters).any(axis=0).nonzero().squeeze(1)
            if self.nonfixed_sig:
                x=torch.stack([self.compute_resid_cost((self.X[ras]-self.centroids[i])**2, self.variances[i]) for i in range(self.nc)], axis=1)
            else:
                x=torch.stack([self.compute_resid_cost((self.X[ras]-self.centroids[i])**2) for i in range(self.nc)], axis=1)
            na = self.clusters.argmax(axis=0)
            breakpoint()
        for i in range(self.nc):
            self.assign_subclusters(i)
            self.update_subclusters(i)
            if torch.logical_and(self.subclusters[i].sum(axis=1)==0, self.clusters[i].sum()>1).any():
                self.init_new_subcentroids(i)
        if not ( (self.clusters.sum(axis=0)==1).all()):
            breakpoint()
        idx = self.subclusters.to(self.X.dtype)
        self.subcentroids = (idx@self.X) / (idx.sum(axis=2)[:,:,None]+1e-8)
        if self.run_checks and self.resid_cost_by_c.sum() > before_resid+1e-10:
            print(f'mod: {before_mod:.3f}->{self.model_cost:.3f}, idx: {before_idx:.3f}->{self.idx_cost.item():.3f}, resid: {before_resid:.3f}->{self.resid_cost_by_c.sum().item():.3f}')
            ras = (self.clusters!=before_clusters).any(axis=0).nonzero().squeeze(1)
            if self.nonfixed_sig:
                x=torch.stack([self.compute_resid_cost((self.X-self.centroids[i])**2, self.variances[i]) for i in range(self.nc)], axis=1)
            else:
                x=torch.stack([self.compute_resid_cost((self.X-self.centroids[i])**2) for i in range(self.nc)], axis=1)
            na = self.clusters.argmax(axis=0)
            oa = before_clusters.argmax(axis=0)
            breakpoint()
        if self.run_checks and self.subclusters[0,0].bool().any() and not ( torch.allclose(self.subcentroids[0,0], self.X[self.subclusters[0,0].bool()].mean(axis=0))):
            breakpoint()
        if self.run_checks and self.subclusters[0,1].bool().any() and not ( torch.allclose(self.subcentroids[0,1], self.X[self.subclusters[0,1].bool()].mean(axis=0))):
            breakpoint()
        if self.run_checks and self.subclusters[0,1].bool().any() and (not (self.subclusters.sum(axis=[0,1])==1).all()):
            breakpoint()

    #@profile_lines
    def maybe_split(self):
        if self.nonfixed_idx_cost:
            subcluster_sizes = self.subclusters.sum(axis=2)
            split_idx_cost = ((self.logn - subcluster_sizes.log()) @ subcluster_sizes.T.float()).diagonal()
            idx_change_from_splitting = split_idx_cost - self.idx_cost_by_c
        else:
            idx_change_from_splitting = self.n * (torch.log(torch.tensor(self.nc+1)) - torch.log(torch.tensor(self.nc))) # harmonic approximation
        resid_cost_by_subc = -torch.ones([self.nc, 2], device=self.X.device)
        subc_bools = self.subclusters.bool()
        for i in range(self.nc):
            if self.clusters[i].sum()==1:
                resid_cost_by_subc[i,:] = torch.inf
            else:
                for j in range(2):
                    cluster_points = self.X[subc_bools[i,j]]
                    squared_dists = ((cluster_points - self.subcentroids[i,j])**2)
                    #sigs = self.subvariances[i,j] if self.nonfixed_sig else torch.ones(self.nz)
                    x = self.compute_resid_cost(squared_dists)
                    #x = 0.5 * (self.nz*LOG2PI + squared_dists.sum(axis=1))
                    assert x.ndim==1
                    #if self.nc == 8:
                        #breakpoint()
                    resid_cost_by_subc[i,j] = x.sum()

        resid_cost_by_subc = resid_cost_by_subc.sum(axis=1)
        resid_change_from_splitting = resid_cost_by_subc - self.resid_cost_by_c
        change_by_c = resid_change_from_splitting + idx_change_from_splitting + self.cluster_cost
        change_by_c = resid_change_from_splitting + idx_change_from_splitting + self.cluster_cost
        #for i in range(len(change_by_c)):
            #print(f'change_by_c[{i}] ({change_by_c[i]:.4f}) = resid_change_from_splitting ({resid_change_from_splitting[i].item():.4f}) + idx_change_from_splitting ({idx_change_from_splitting.item():.4f}) + self.cluster_cost ({self.cluster_cost.item():.4f})')
        if not ( change_by_c.ndim==1 and change_by_c.shape[0]==self.nc):
            breakpoint()
        if self.nc==1:
            i, best_savings = 0, change_by_c
        else:
            best_savings, i = change_by_c.min(dim=0)
        if best_savings < 0:
            self.split(i)
            return True
        else:
            #if self.run_checks and self.mdl_cost() > before:
                #breakpoint()
            return False

    def split(self, i):
        if self.run_checks:
            before = self.mdl_cost()
            before_mod = self.model_cost
            before_idx = self.idx_cost.item()
            before_resid = self.resid_cost_by_c#.sum().item()
            before_clusters = self.clusters.clone()
            before_subclusters = self.subclusters.clone()
        self.clusters[i] = self.subclusters[i,0]
        self.clusters = torch.cat([self.clusters, self.subclusters[i,1][None]])
        self.centroids[i] = self.subcentroids[i,0]
        self.variances[i] = self.subvariances[i,0]
        self.centroids = torch.cat([self.centroids, self.subcentroids[i,1][None]])
        self.variances = torch.cat([self.variances, self.subvariances[i,1][None]])
        self.init_new_subcentroids(i)
        self.subclusters = torch.cat([self.subclusters, torch.zeros(1, 2, self.n, device=self.device)])
        self.subcentroids = torch.cat([self.subcentroids, torch.zeros(1, 2, self.nz, device=self.device)])
        self.subvariances = torch.cat([self.subvariances, torch.zeros(1, 2, self.nz, device=self.device)])
        if not ( (self.subclusters.sum(axis=[0,1])<=1).all()):
            breakpoint()
        self.init_new_subcentroids(self.nc-1)
        if not ( torch.logical_or(self.subclusters.sum(axis=2).all(axis=1), self.clusters.sum(axis=1)==1).all()):
            breakpoint()
        if self.run_checks and self.mdl_cost() >= before:
            breakpoint()

    def maybe_split_dist_diffs(self):
        self.cluster_dists = torch.cdist(self.X, self.centroids)
        self.subcluster_dists = torch.cdist(self.X, self.subcentroids.flatten(0, 1)).reshape(self.n, self.nc, 2)
        curr_score = self.slow_dist_diffs(self.centroids, self.clusters)
        #curr_score = 0 if self.nc==1 else silhouette_score(self.X, self.clusters.argmax(axis=0))
        #curr_score = 0 if self.nc==1 else 1/davies_bouldin_score(self.X, self.clusters.argmax(axis=0))
        best_score = curr_score
        idx_to_split = -1
        scores_list = []
        for i in range(self.nc):
            centroids_if_split = torch.cat([self.centroids[:i], self.subcentroids[i], self.centroids[i+1:]])
            clusters_if_split = torch.cat([self.clusters[:i], self.subclusters[i], self.clusters[i+1:]])
            score = self.slow_dist_diffs(centroids_if_split, clusters_if_split)
            #score = 1/davies_bouldin_score(self.X, clusters_if_split.argmax(axis=0))
            if score > best_score:
                best_score = score
                idx_to_split = i
            scores_list.append(score)
        #if self.nc > 4:
            #breakpoint()
        if idx_to_split >= 0:
            if self.verbose:
                print(f'current: {curr_score:.5f}, split-scores:', ' '.join(f'{x:.5f}' for x in scores_list))
                print(f'splitting at {idx_to_split}')
            self.split(i)
            return True
        else:
            return False

    def slow_dist_diffs(self, centroids, clusters):
        """Part of alternative I explored that maximises the difference between
        mean-dist-of-point-to-it's-centroid and mean-dist-between-centroids"""
        clusters = clusters.bool()
        C = (torch.cdist(centroids, centroids)**0.5).mean()
        dlist = []
        for clst, cent in zip(clusters, centroids):
            cluster_vecs = self.X[clst]
            dlist.append((((cluster_vecs-cent)**2).sum(axis=1)**0.5).sum())
        D = sum(dlist)
        return C - D

    #@profile_lines
    def maybe_merge(self):
        if self.run_checks:
            before = self.mdl_cost()
        cluster_sizes = self.clusters.sum(axis=1)
        #mn_centroids = torch.zeros(self.nc, self.nc, self.nz, device=self.device)
        #mn_vars = torch.zeros(self.nc, self.nc, self.nz, device=self.device)

        centroid_dists = torch.cdist(self.centroids, self.centroids, p=2)
        centroid_dists[torch.eye(self.nc).bool()] = torch.inf
        am = centroid_dists.argmin()
        midx1 = am // self.nc
        midx2 = am % self.nc
        n1, n2 = cluster_sizes[midx1], cluster_sizes[midx2]
        mn_cent = (self.centroids[midx1]*n1 + self.centroids[midx2]*n2) / (n1+n2)
        mn_cluster = torch.logical_or(self.clusters[midx1].bool(), self.clusters[midx2].bool())
        assert mn_cluster.sum() == n1+n2
        mn_points = self.X[mn_cluster]
        cur_resid_cost = self.resid_cost_by_c[midx1] + self.resid_cost_by_c[midx2]
        mn_sq_dists = (mn_points - mn_cent)**2
        mn_var = ((self.variances[midx1] + self.variances[midx2]) / 2) + (self.centroids[midx1] - self.centroids[midx2])**2 # EVE's law, though stricly should weight
        mn_resid_cost = self.compute_resid_cost(mn_sq_dists, mn_var if self.nonfixed_sig else None)
        resid_increase_from_merging = mn_resid_cost.sum() - cur_resid_cost
        idx_decrease_from_merging = self.n * (torch.log(torch.tensor(self.nc)) - torch.log(torch.tensor(self.nc-1))) # harmonic approximation
        single_merge_savings = idx_decrease_from_merging - resid_increase_from_merging + self.cluster_cost

        if single_merge_savings > 0:
            #if not ( midx1 < midx2): # shouldn't have come from lower triangle):
                #breakpoint()
            if self.run_checks:
                before_cost = self.mdl_cost()
                before_mod = self.model_cost
                before_idx = self.idx_cost.item()
                before_resid = self.resid_cost_by_c.clone()
                before_clusters = self.clusters.clone()
                before_subclusters = self.subclusters.clone()
            self.subclusters[midx1,0] = self.clusters[midx1]
            self.subclusters[midx1,1] = self.clusters[midx2]
            self.subcentroids[midx1,0] = self.centroids[midx1]
            self.subcentroids[midx1,1] = self.centroids[midx2]
            self.clusters[midx1] += self.clusters[midx2]
            self.centroids[midx1] = mn_cent
            self.variances[midx1] = mn_var
            self.clusters = torch.cat([self.clusters[:midx2], self.clusters[midx2+1:]])
            self.subclusters = torch.cat([self.subclusters[:midx2], self.subclusters[midx2+1:]])
            self.centroids = torch.cat([self.centroids[:midx2], self.centroids[midx2+1:]])
            self.subcentroids = torch.cat([self.subcentroids[:midx2], self.subcentroids[midx2+1:]])
            self.variances = torch.cat([self.variances[:midx2], self.variances[midx2+1:]])
            self.subvariances = torch.cat([self.subvariances[:midx2], self.subvariances[midx2+1:]])
            #print(f'merged {midx1} and {midx2}, now have {self.nc} clusters')
            if self.run_checks:
                print(f'merging, going down from {before} to {self.mdl_cost()}')
                if self.mdl_cost() >= before:
                    breakpoint()
            return True
        else:
            return False

    #@profile_lines
    def fit_predict(self, X, seed_k=1, return_ks=False, patience=5):
        starttime = time()
        with torch.no_grad():
            assert X.ndim == 2
            self.n, self.nz = X.shape
            self.logn = np.log(self.n)
            self.cluster_cost = compute_cluster_cost(X)
            if self.nonfixed_sig:
                self.cluster_cost *= 2

            self.X = torch.tensor(X, device=self.device, dtype=torch.float64)
            st = time()
            if self.subinit=='ksm':
                if self.compute_full_densities:
                    #self.point_densities = (-(self.X - self.X[:,None])**2).sum(axis=2).exp().mean(axis=1)
                    distances = torch.cdist(self.X, self.X, p=2)
                    self.point_densities = (-distances).exp().mean(dim=1)
                else:
                    self.point_densities = lattice_density(self.X)
            else:
                self.point_densities = None
            best_overall_cost = torch.inf
            for attempt in range(self.n_init):
                if seed_k==1:
                    self.clusters = torch.ones([1,self.n], device=self.device).long()
                    self.subclusters = torch.ones([1,2,self.n], device=self.device).long()
                    self.centroids = self.X.mean(axis=0, keepdim=True)
                else:
                    km = KMeans(seed_k)
                    clabels = torch.tensor(km.fit_predict(X))
                    self.clusters = ((torch.arange(seed_k).repeat(self.n, 1).T == clabels).long())
                    self.centroids = torch.tensor(km.cluster_centers_).double()
                    self.subclusters = torch.empty([seed_k,2,self.n], device=self.device).long()
                assert not self.nonfixed_sig, 'havent implemented initing the variances'
                self.variances = torch.ones(seed_k, self.nz, device=self.device)

                all_init_subcs, all_init_subvars = [], []
                for i in range(seed_k):
                    init_subc, init_subvar = self.new_subcentroid(0)
                    all_init_subcs.append(init_subc)
                    all_init_subvars.append(init_subvar)
                self.subcentroids = torch.stack(all_init_subcs)
                self.subvariances = torch.stack(all_init_subvars)
                assert self.subvariances.shape == self.subcentroids.shape
                self.init_new_subcentroids(0)
                best_cost = self.mdl_cost().item()
                tol = 2
                self.best_resid_subc_cost = torch.inf
                unimproved_count = 0
                tot_kms = 0; tot_split = 0
                all_pred_ks = []
                while True:
                    all_pred_ks.append(self.nc)
                    before = self.centroids.clone()
                    if self.verbose:
                        print(f'Num Clusters: {self.nc}  Cost: {self.mdl_cost():.15f}  count: {unimproved_count}')
                    kms_starttime = time()
                    self.kmeans_step()
                    tot_kms += time()-kms_starttime
                    split_starttime = time()
                    if self.dist_diffs:
                        did_split = self.maybe_split_dist_diffs()
                    else:
                        did_split = self.maybe_split()
                    tot_split += time()-split_starttime
                    if did_split:
                        #self.kmeans_step()
                        pass
                    elif self.nc>1:
                        did_merge = self.maybe_merge()
                        #if did_merge:
                            #self.kmeans_step()
                    new_cost = self.mdl_cost()
                    if new_cost<best_cost-tol:
                        best_cost = new_cost
                        unimproved_count = 0
                    else:
                        unimproved_count += 1
                    if unimproved_count == patience:
                        break
                    if (self.centroids.shape==before.shape) and (self.centroids == before).all():
                        break
                assert (self.clusters.sum(axis=0)==1).all()

                #if self.mdl_cost() < best_overall_cost:
                    #best_cluster_labels = self.clusters.argmax(axis=0).detach().cpu().numpy()
                    #best_overall_cost = self.mdl_cost()
            if self.verbose:
                print(f'kms: {tot_kms:.5f}, split: {tot_split:.5f}, tot: {time()-starttime:.5f}', self.device)
            best_cluster_labels = self.clusters.argmax(axis=0).detach().cpu().numpy()
            if return_ks:
                return best_cluster_labels, all_pred_ks
            else:
                return best_cluster_labels

def acc(preds, gts):
    acc1 = accuracy(preds, gts)
    acc2 = accuracy(gts, preds)
    score = (2*acc1*acc2) / (acc1+acc2)
    return score

def lattice_density(X, grid_size=0.1):
    """
    Compute density using a grid-based approach:
    1. Discretize points into grid cells
    2. Count points per cell
    3. Transfer counts back to points
    """
    # Discretize points into grid indices
    grid_indices = (X / grid_size).long()

    # Get unique grid cells and their counts
    unique_cells, points_per_cell = torch.unique(grid_indices, dim=0, return_counts=True)

    # Create lookup dictionary for cell counts
    cell_counts = {tuple(cell.tolist()): count.item()
                  for cell, count in zip(unique_cells, points_per_cell)}

    # Map each point to its cell's count
    densities = torch.tensor([cell_counts[tuple(idx.tolist())]
                            for idx in grid_indices], device=X.device)

    return densities / len(X)  # Normalize

def lablled_cluster_metrics(preds, gt_point_labs):
    results = {}
    assert (len(preds)==0) == (len(gt_point_labs)==0)
    for mname in ['acc', 'ari', 'nmi']:
        if len(preds)==0:
            score = -1
        else:
            mfunc = globals()[mname]
            score = mfunc(gt_point_labs, preds)
        results[mname] = score
    return results

def unlabelled_cluster_metrics(data, preds):
    data = data.reshape(data.shape[0], -1)
    results = {'silhouette': silhouette_score(data, preds),
               'davies-bouldin': davies_bouldin_score(data, preds)}
    return results


if __name__ == '__main__':
    from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, SpectralClustering, MeanShift, AffinityPropagation, OPTICS
    import matplotlib.pyplot as plt
    import pandas as pd
    from get_dsets import load_mnist, load_usps, load_speech_commands, load_all_in_tree, load_ng_feats, load_im_feats, load_msrvtt, load_pbmc, maybe_cached_dimred
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
    from sklearn.datasets import load_iris, load_wine
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from xmeans import DummyXMeans
    from sweepkm import KMeansBICSweep, ElbowMethod
    from baselines import CRP, DivisiveHierarchical, Pymc3DPMM, DensityPeaksClustering, KMeansMDLSweep, QuickShift
    from natsort import natsorted
    import argparse
    from memory_manager_wrapper import MemoryManagerWrapper
    from dl_utils.label_funcs import compress_labels


    parser = argparse.ArgumentParser()
    parser.add_argument('--compute-full-densities', action='store_true', help='for the very first step, actually compute how close every point is to others, much slower and no more accurate really')
    parser.add_argument('--compute-outliers', action='store_true')
    parser.add_argument('--dim-red-alg', type=str, default='umap')
    parser.add_argument('--dist-diffs', action='store_true')
    parser.add_argument('--dset', '-d', type=str, default='mnist')
    parser.add_argument('--incl-unlabelled-metrics', action='store_true')
    parser.add_argument('--methods', type=str, nargs='+', default=['all'])
    parser.add_argument('--include-slow-baselines', action='store_true')
    parser.add_argument('--metrics-type', type=str, default='labelled')
    parser.add_argument('--ninit', type=int, default=1)
    parser.add_argument('--nonfixed-idx-cost', action='store_true')
    parser.add_argument('--nonfixed-sig', action='store_true')
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--radd', type=int, default=0)
    parser.add_argument('--recompute-dim-red', action='store_true')
    parser.add_argument('--recompute-msrvtt-feats', action='store_true')
    parser.add_argument('--run-checks', action='store_true')
    parser.add_argument('--show-plots', action='store_true')
    parser.add_argument('--subinit', type=str, default='ksm')
    parser.add_argument('--verbose', action='store_true')
    ARGS = parser.parse_args()
    if ARGS.dset=='mnist':
        X, gtlabels = load_mnist(split='both')
    elif ARGS.dset=='usps':
        X, gtlabels = load_usps()
    elif ARGS.dset=='im':
        X, allpaths = load_all_in_tree('data/imagenette2-160')
        imagenet_ids = [x.split('/')[-2] for x in allpaths]
        assert all(x.startswith('n') for x in imagenet_ids)
        unique_ids = list(set(imagenet_ids))
        gtlabels = np.array([unique_ids.index(y) for y in imagenet_ids])
    elif ARGS.dset=='ng20':
        X, gtlabels = load_ng_feats()
    elif ARGS.dset=='im-feats':
        X, gtlabels = load_im_feats()
    elif ARGS.dset=='iris':
        iris = load_iris()
        X, gtlabels = iris.data, iris.target
    elif ARGS.dset=='wine':
        wine = load_wine()
        X, gtlabels = wine.data, wine.target
    elif ARGS.dset=='pendigits':
        pendigits = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra", header=None).values
        X, gtlabels = pendigits[:, :-1], LabelEncoder().fit_transform(pendigits[:, -1])
    elif ARGS.dset=='speech-commands':
        X, gtlabels = load_speech_commands()
    elif ARGS.dset=='msrvtt-iv':
        X, gtlabels = load_msrvtt('iv', ARGS.recompute_msrvtt_feats)
    elif ARGS.dset=='msrvtt-clip':
        X, gtlabels = load_msrvtt('clip', ARGS.recompute_msrvtt_feats)
    elif ARGS.dset=='audioset':
        all_fps = [os.path.join(dirname, fn) for dirname, _, filenames in os.walk('data/audioset_v1_embedding_pytorch') for fn in filenames]
        all_fps = natsorted(all_fps)
        breakpoint()
        X = torch.stack([torch.load(fp) for fp in all_fps])
        breakpoint()
        gtlabels = None
    elif ARGS.dset=='pbmc':
        X = load_pbmc()
        gtlabels = None
    else:
        print(f'Unrecognised dataset: {ARGS.dset}')

    dim_red_X = maybe_cached_dimred(ARGS.dset, X, ARGS.dim_red_alg, recompute=ARGS.recompute_dim_red)
    if ARGS.radd > 0:
        raddition = np.random.rand(*dim_red_X[:ARGS.radd].shape)
        raddition *= (dim_red_X.max(axis=0) - dim_red_X.min(axis=0))
        raddition += dim_red_X.min(axis=0)# -1000
        dim_red_X = np.append(dim_red_X, raddition,axis=0)
    models = {}
    if 'ksm' in ARGS.methods or 'all' in ARGS.methods:
        models['ksm'] = KStarMeans(n_init=ARGS.ninit, nonfixed_idx_cost=ARGS.nonfixed_idx_cost, nonfixed_sig=ARGS.nonfixed_sig, dist_diffs=ARGS.dist_diffs, compute_full_densities=ARGS.compute_full_densities, subinit=ARGS.subinit, verbose=ARGS.verbose, run_checks=ARGS.run_checks, compute_outliers=ARGS.compute_outliers)
    if 'kmbicsweep' in ARGS.methods or 'all' in ARGS.methods:
        models['kmbicsweep'] = KMeansBICSweep()
    if 'dpc' in ARGS.methods or 'all' in ARGS.methods:
        models['dpc'] = DensityPeaksClustering()
    if 'dbscan' in ARGS.methods or 'all' in ARGS.methods:
        models['dbscan'] = DBSCAN()
    if 'hdbscan' in ARGS.methods or 'all' in ARGS.methods:
        models['hdbscan'] = HDBSCAN()
    if 'xmeans' in ARGS.methods or 'all' in ARGS.methods:
        models['xmeans'] = DummyXMeans(kmax=int(len(X)**0.5))
    if 'elbow' in ARGS.methods or 'all' in ARGS.methods:
        models['elbow'] = ElbowMethod()
    if 'dpmm' in ARGS.methods or 'all' in ARGS.methods:
        #models['dpmm'] = BayesianGaussianMixture(n_components=int(len(X)**0.5), init_params='k-means++', covariance_type='spherical', weight_concentration_prior=1.)
        models['dpmm'] = BayesianGaussianMixture(n_components=int(len(X)**0.5), init_params='kmeans', covariance_type='spherical', weight_concentration_prior=1.)
        #models['dpmm'] = Pymc3DPMM()
    if 'crp' in ARGS.methods or 'all' in ARGS.methods:
        models['crp'] = CRP()
    if 'divhier' in ARGS.methods or 'all' in ARGS.methods:
        models['divhier'] = DivisiveHierarchical(kmax=int(len(X)**0.5))
    if 'kmmdlsweep' in ARGS.methods or 'all' in ARGS.methods:
        models['kmmdlsweep'] = KMeansMDLSweep()
    if 'qs' in ARGS.methods or 'all' in ARGS.methods:
        models['qs'] = QuickShift()

    if gtlabels is not None:
        gt_nc = len(set(gtlabels))
        models['kmeans'] = KMeans(n_clusters=gt_nc, n_init=ARGS.ninit)
        models['gmm'] = GaussianMixture(n_components=gt_nc, n_init=ARGS.ninit)
    if ARGS.include_slow_baselines:
        models['optics'] = OPTICS(min_samples=2)
        spectral = SpectralClustering(n_clusters=gt_nc, n_init=ARGS.ninit)
        models['spectral'] = MemoryManagerWrapper(spectral)
        meanshift = MeanShift(bandwidth=np.std(dim_red_X) * 0.5)
        models['meanshift'] = MemoryManagerWrapper(meanshift)
        affinity = AffinityPropagation(damping=0.9, max_iter=200)
        models['affinity'] = MemoryManagerWrapper(affinity)


    all_results = {}
    all_preds = {}
    def cluster_metrics(X_full, X_dimred, preds, gts, include_unlabelled_metrics):
        if include_unlabelled_metrics:
            results = unlabelled_cluster_metrics(X_dimred, preds)
            r_full = unlabelled_cluster_metrics(X_full, preds)
            results.update({f'{k}-full':v for k,v in r_full.items()})
        else:
            results = {}
        if gtlabels is not None:
            r2 = lablled_cluster_metrics(gtlabels[inliers_mask], preds)
            results.update(r2)
        results = {k:100*v for k,v in results.items()}
        return results

    metrics_list = [] if gtlabels is None else ['acc', 'ari', 'nmi', 'nc', 'n_outliers', 'runtime']
    if ARGS.incl_unlabelled_metrics:
        metrics_list += ['silhouette', 'davies_bouldin']
    #all_pred_names = ['ksm', 'kmeans', 'gmm', 'dbscan', 'hdbscan', 'xmeans']
    stochastic_methods = ['kmbicsweep', 'ksm', 'kmeans', 'gmm', 'xmeans', 'hdbscan', 'elbow', 'crp', 'dpmm', 'divhier', 'dpc', 'kmmdlsweep', 'qs']
    for pred_name, model in models.items():
        model = models[pred_name]
        if ARGS.methods != ['all'] and pred_name not in ARGS.methods:
            continue
        print(pred_name)
        ntrials = ARGS.ntrials if pred_name in stochastic_methods else 1
        all_trial_results = []
        for _ in range(ntrials):
            print(_)
            starttime = time()
            preds = 'OOM' if pred_name=='spectral' and ARGS.dset in ['mnist', 'speech-commands', 'msrvtt-clip', 'msrvtt-iv'] else model.fit_predict(dim_red_X)
            if isinstance(preds, str):
                assert preds == 'OOM'
                print('OOM')
                all_trial_results.append({k:-1 for k in metrics_list})
                continue
            n_outliers = (preds==-1).sum()
            runtime = time()-starttime
            inliers_mask = preds!=-1
            drX = dim_red_X[inliers_mask]
            preds = preds[inliers_mask]
            masked_gtlabels = None if gtlabels is None else gtlabels[inliers_mask]
            results = cluster_metrics(X[inliers_mask], drX, preds, masked_gtlabels, include_unlabelled_metrics=ARGS.incl_unlabelled_metrics)
            assert -1 not in preds
            results['nc'] = len(set(preds))# - (1 if -1 in preds else 0)
            results['n_outliers'] = n_outliers
            results['runtime'] = runtime
            all_trial_results.append(results)
        df = pd.DataFrame(all_trial_results)
        results_mean = df.mean(axis=0)
        results_std = np.zeros_like(results_mean.values) if ntrials==1 else df.std(axis=0).values
        all_results[pred_name] = {k:f'{m:.2f} ({s:.4f})' for k,m,s in zip(df.columns, results_mean.values, results_std)}
        all_preds[pred_name] = preds

    df = pd.DataFrame(all_results).T
    print(df)
    import matplotlib.colors as mcolors

    def plot_with_colours(predname, labels):
        classes = np.unique(labels)
        cmap = plt.get_cmap('hsv', len(classes))  # Discrete colormap
        norm = mcolors.BoundaryNorm(np.arange(len(classes) + 1) - 0.5, len(classes))

        d = np.random.permutation(np.arange(len(set(labels))))
        labels, _, _ = compress_labels(labels)
        labels = [d[x] for x in labels]
        sc = plt.scatter(dim_red_X[:, 0], dim_red_X[:, 1], c=labels, cmap=cmap, norm=norm)
        #plt.colorbar(sc, ticks=classes, shrink=0.8, aspect=10)
        cb = plt.colorbar(sc, shrink=0.8, aspect=10)
        if len(set(classes)) < 50:
            cb.set_ticks(classes)
            cb.set_ticklabels([str(i) for i in classes])
        else:
            cb.set_ticks([])

        plt.savefig(save_fp:=f'results/scatter_plots/{predname}-{ARGS.dset}-scatter.png')
        plt.clf()
        if ARGS.show_plots:
            os.system(f'/usr/bin/xdg-open {save_fp}')

    expname = '{ARGS.dset}-{ARGS.ntrials}-{"".join(ARGS.methods)}'
    if 'ksm' in all_preds.keys():
        plot_with_colours('ours', all_preds['ksm'])
    for method_name, method_preds in all_preds.items():
        plot_with_colours(method_name, method_preds)
    #if gtlabels is not None:
        #plot_with_colours('gt', np.concatenate([gtlabels, -np.ones(ARGS.radd)]))
    results_fp = f'results/labelled_dsets/{expname}trials.csv' if ARGS.include_slow_baselines else f'results/labelled_dsets/{expname}trials_noslows.csv'
    os.makedirs(os.path.dirname(results_fp), exist_ok=True)
    df.to_csv(results_fp)

    def reformat_line(line):
        parts = line.split()
        result = []
        i = 0
        while i < len(parts):
            try:
                mean = float(parts[i])
                std = float(parts[i+1].strip("()"))
                if std == 0:
                    result.append(f"{mean:.2f}")
                else:
                    result.append(f"{mean:.2f} ({std:.3f})")
                i += 2
            except ValueError:
                result.append(parts[i])
                i += 1
        return ' & '.join(result)

    for mn in df.index:
        joined = ' & '.join(df.loc[mn].values)
        print(f'& {mn.upper()} & {joined} \\\\'.replace('(0.0000) ', ''))
