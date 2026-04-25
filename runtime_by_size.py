from numpy.lib.stride_tricks import sliding_window_view
import os
import numpy as np
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kstar_means import KStarMeans, maybe_cached_dimred
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, SpectralClustering, MeanShift, AffinityPropagation, OPTICS
from get_dsets import load_speech_commands
from xmeans import DummyXMeans
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--recompute', action='store_true')
parser.add_argument('--include-slow-baselines', action='store_true')
parser.add_argument('--step-size', type=int, default=1000)
parser.add_argument('--n-trials', type=int, default=10)
ARGS = parser.parse_args()



os.makedirs('results/runtimes', exist_ok=True)
breakpoint()
if os.path.exists(csv_fp:=f"results/runtimes/clustering_runtime-{ARGS.step_size}-ntrials{ARGS.n_trials}-slows{ARGS.include_slow_baselines}.csv") and not ARGS.recompute:
    df = pd.read_csv(csv_fp)
else:
    X, gtlabels = load_speech_commands()
    dim_red_X = maybe_cached_dimred('speech-commands', X, 'umap', recompute=False)
    gt_nc = len(set(gtlabels))
    method_dict = {}
    method_dict['KStarMeans'] = KStarMeans(n_init=1, nonfixed_idx_cost=False, nonfixed_sig=False, dist_diffs=False, compute_full_densities=False, subinit='kplusm', verbose=False, run_checks=False, compute_outliers=False)
    method_dict['KMeans'] = KMeans(n_clusters=gt_nc, n_init=1)
    method_dict['DBSCAN'] = DBSCAN()
    method_dict['HDBSCAN'] = HDBSCAN()
    method_dict['XMeans'] = DummyXMeans(kmax=int(len(X)**0.5))
    method_dict['GMM'] = GaussianMixture(n_components=gt_nc, n_init=1)
    if ARGS.include_slow_baselines:
        method_dict['Spectral'] = SpectralClustering(n_clusters=gt_nc, n_init=1)
    results = {}
    #power = 1.05
    #n_incrs = math.ceil(np.log(len(dim_red_X)) / np.log(power))
    #sample_sizes = [int(start_size*power**i) for i in range(n_incrs)]
    sample_sizes = range(ARGS.step_size, len(dim_red_X), ARGS.step_size)
    breakpoint()
    for N in sample_sizes:
        #X_sub = dim_red_X[:N]

        N_results = {}
        n_trials = 10
        #X_subs = [dim_red_X[np.random.choice(len(dim_red_X), N, replace=False)] for _ in range (n_trials)]
        X_subs = []
        for _ in range(n_trials):
            #start = np.random.choice(len(dim_red_X))
            #start = np.random.choice([0, 1000, 3000, 10000, 30000])
            start = np.random.choice(np.arange(0, len(dim_red_X), 3000))
            end = start+N
            first_chunk = dim_red_X[start:end]
            second_chunk = dim_red_X[:max(0, end-len(dim_red_X))]
            X_subs.append(np.concatenate([first_chunk, second_chunk]))
        for mod_name, mod in method_dict.items():
            start = time.time()
            for i in range(n_trials):
                mod.fit_predict(X_subs[i])
            N_results[mod_name] = (time.time() - start) / n_trials

        results[N] = N_results
        print(N, N_results)

    # Save to CSV
    df = pd.DataFrame(results).T
    df.to_csv(csv_fp, index=False)
df.index.name = 'N'
plot_df = df.reset_index().melt(id_vars='N', var_name='method', value_name='time')

# Plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
smoothed = []
window_size = 10
for i in range(len(df)):
    start = int(max(0, i-math.floor(window_size/2)))
    end = int(min(len(df), i+math.ceil(window_size/2)))
    window = df.values[start:end]
    weights = np.array([1.]) if len(window)==1 else np.sin(np.pi * np.linspace(0, 1, window.shape[0]))
    weights /= weights.sum()
    smoothed.append(weights@window)
df_windowed = pd.DataFrame(smoothed, index=df.index, columns=df.columns)
df_windowed.index.name = df.index.name
plot_df_windowed = df_windowed.reset_index().melt(id_vars='N', var_name='method', value_name='time')
palette = {'KStarMeans': 'green', 'KMeans': 'red', 'GMM': 'yellow', 'DBSCAN': 'orange', 'HDBSCAN': 'purple', 'XMeans': 'blue'}
sns.lineplot(data=plot_df_windowed, x='N', y='time', hue='method', lw=2, palette=palette)
sns.scatterplot(data=plot_df, x='N', y='time', hue='method', marker='o', legend=False, palette=palette)
plt.title('Runtime vs Dataset Size')
plt.xlabel('Number of Samples')
plt.ylabel('Time (s)')
plt.tight_layout()
fig_fp = csv_fp.replace('.csv', '.png')
plt.savefig(fig_fp)
os.system(f'/usr/bin/xdg-open {fig_fp}')

