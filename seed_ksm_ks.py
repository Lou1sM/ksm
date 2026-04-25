import os
from kstar_means import KStarMeans, maybe_cached_dimred
import matplotlib.pyplot as plt
from get_dsets import load_mnist, load_ng_feats, load_speech_commands, load_msrvtt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
import torchvision.datasets as tdatasets
from collections import defaultdict
from time import time


def seed_ks(X, gtn):
    all_preds = []
    all_times = {}
    for k in np.arange(max(1, gtn-20), gtn+10):
        ksm = KStarMeans(subinit='kplusm')
        starttime = time()
        ntrials = 10
        for trial in range(ntrials):
            _, evolution_of_k = ksm.fit_predict(X, seed_k=k, return_ks=True, patience=5)
        runtime = (time()-starttime) / ntrials
        all_preds.append(evolution_of_k)
        all_times[k] = runtime
    return all_preds, all_times

import ssl
X_by_dset_name = {}
y_by_dset_name = {}
ssl._create_default_https_context = ssl._create_unverified_context
dtrain=tdatasets.USPS(root='data/usps',train=True,download=True)
dtest=tdatasets.USPS(root='data/usps',train=False,download=True)
X_by_dset_name['usps'] = np.concatenate([dtrain.data,dtest.data])
y_by_dset_name['usps'] = np.concatenate([dtrain.targets,dtest.targets])
X_by_dset_name['mnist'], y_by_dset_name['mnist'] = load_mnist(split='both')
X_by_dset_name['20ng'], y_by_dset_name['20ng'] = load_ng_feats()
X_by_dset_name['msrvtt'], y_by_dset_name['msrvtt'] = load_msrvtt('clip', recompute_feats=False)
X_by_dset_name['sc'], y_by_dset_name['sc'] = load_speech_commands()
ks_by_dset_name = {}
times_by_dset_name = {}
for dset_name in X_by_dset_name.keys():
    dimred_X = maybe_cached_dimred(dset_name, X_by_dset_name[dset_name], 'umap')
    ks_by_dset_name[dset_name], times_by_dset_name[dset_name] = seed_ks(dimred_X, len(set(y_by_dset_name[dset_name])))


target_len = 100  # or whatever length you want
data = []

for dset_name, list_of_lists in ks_by_dset_name.items():

    final_ks = []
    for series in list_of_lists:
        x_old = np.linspace(0, 1, len(series))
        x_new = np.linspace(0, 1, target_len)
        f = interp1d(x_old, series, kind='linear')
        resampled = f(x_new)
        data.append(pd.DataFrame({'x': range(target_len), 'y': resampled, 'label': dset_name}))
        final_ks.append(series[-1])

    mean_pred_k = np.mean(final_ks)
    for x in data:
        sns.lineplot(data=x, x='x', y='y', color='blue')
    plt.axhline(y=8, linestyle='--', color='black')
    plt.text(x=max(x['x'])*1.05, y=mean_pred_k + 0.1, s=f'{mean_pred_k:.2f}', color='black', ha='right', va='bottom', fontsize=12)
    plt.xlabel('Train Step (normalized)')
    plt.ylabel('k')
    plt.title('Evolution of k for Different Initial k')
    plt.savefig(out_fp:=f'results/seeded_k_vs_predk{dset_name}.png')
    os.system(f'/usr/bin/xdg-open {out_fp}')
    plt.clf()

    runtime_by_k = times_by_dset_name[dset_name]
    ks = list(runtime_by_k.keys())
    runtimes = list(runtime_by_k.values())

    plt.plot(ks, runtimes, marker='o')
    gtk = len(set(y_by_dset_name[dset_name]))
    plt.axvline(x=gtk, linestyle='--', color='red')
    plt.xlabel('Seeded k')
    plt.ylabel('Runtime')
    plt.title('Runtime vs Initial k')
    plt.savefig(out_fp:=f'results/seeded_k_vs_runtime{dset_name}.png')
    os.system(f'/usr/bin/xdg-open {out_fp}')
    plt.clf()

    breakpoint()


