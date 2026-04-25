import os
from matplotlib.ticker import FuncFormatter
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, SpectralClustering, MeanShift, AffinityPropagation, OPTICS
from get_dsets import load_mnist, load_speech_commands, load_all_in_tree, load_ng_feats, load_im_feats, load_msrvtt
from kstar_means import maybe_cached_dimred
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


n_eps = 80
n_mpts = 40
ks = -np.ones([n_eps, n_mpts])
eps_to_sweep = [0.03*1.05**i for i in range(n_eps)]
minpts_to_sweep = np.arange(1,n_mpts+1)

if os.path.exists(np_save_fp:='dbscan-varied-eps-minpts.npy'):
    ks = np.load(np_save_fp)
else:
    X, gtlabels = load_mnist(split='both')
    dim_red_X = maybe_cached_dimred('mnist', X, 'umap')
    for i, eps in enumerate(tqdm(eps_to_sweep)):
        for j, min_samples in enumerate(minpts_to_sweep):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            preds = dbscan.fit_predict(dim_red_X)
            ks[i,j] = len(set(preds)) - (1 if -1 in preds else 0)

    np.save(np_save_fp, ks)

log_ks = np.log(ks)
annot = np.full(ks.shape, '', dtype=object)
for i in range(0, ks.shape[0], 5):
    for j in range(0, ks.shape[1], 5):
        annot[i, j] = annot[i, j] = f'{int(round(ks[i, j]))}'

log_positions = np.geomspace(eps_to_sweep[0], eps_to_sweep[-1], num=5)
yticks = [np.argmin(np.abs(eps_to_sweep - val)) for val in log_positions]
yticklabels = [f'{val:.3f}' for val in log_positions]

ax = sns.heatmap(log_ks, xticklabels=minpts_to_sweep, yticklabels=yticklabels, cmap='viridis',
                 annot=annot, fmt='', annot_kws={"size": 9}, cbar=True)

xticks = np.arange(0, len(minpts_to_sweep), 5) + 0.5
ax.set_xticks(xticks)
xticklabels = [minpts_to_sweep[i] for i in range(0, len(minpts_to_sweep), 5)]
ax.set_xticklabels(xticklabels, rotation=0)
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

for text in ax.texts:
    # Check if this text is in the first row
    if text.get_position()[1] == 0.5:  # Y-position for first row
        text.set_color('black')

ax.yaxis.set_tick_params(pad=10)  # Increase padding

cbar = ax.collections[0].colorbar
cbar.set_label('found k')
cbar.formatter = FuncFormatter(lambda val, pos: f'{int(round(np.exp(val)))}')
cbar.update_ticks()

plt.xlabel('min-pts')
plt.ylabel('eps')

#from matplotlib.colors import LogNorm
#from matplotlib.ticker import FuncFormatter
#fig, ax = plt.subplots()
#
## Plot using imshow with log-normalized colors
#im = ax.imshow(
#    ks,
#    cmap='viridis',
#    norm=LogNorm(),
#    extent=[
#        0, len(minpts_to_sweep),  # x-axis (min-pts)
#        eps_to_sweep[0], eps_to_sweep[-1]  # y-axis (eps)
#    ],
#    aspect='auto'
#)
#
## Annotate selected squares
#for i in range((ks.shape[0]-1)%5, ks.shape[0], 5):
#    for j in range(0, ks.shape[1], 5):
#        #eps_val = eps_to_sweep[i] if i==len(eps_to_sweep)-1 else (eps_to_sweep[i] + eps_to_sweep[i+1])/2
#        eps_val = eps_to_sweep[i] - 0.01
#        minpts_val = j + 1.8  # center of the cell
#        c = 'black' if i==ks.shape[0]-1  or (i>=ks.shape[0]-5 and j==ks.shapep[1]-1) else 'white'
#        to_write = f'{int(round(ks[-i-1, -j-1]))}'
#        if int(to_write)==20:
#            breakpoint()
#        print(eps_val, minpts_val, to_write)
#        ax.text(
#            minpts_val,
#            #eps_to_sweep[-1] - eps_val,
#            eps_val,
#            to_write,
#            ha='center', va='center', fontsize=9, color=c
#        )
#
#xticks = np.arange(0, len(minpts_to_sweep), 5) + 0.5
#ax.set_xticks(xticks)
#ax.set_xticklabels([minpts_to_sweep[i] for i in range(0, len(minpts_to_sweep), 5)])
#
## Set y-axis to log scale with auto ticks
#ax.set_yscale('log')
#ax.yaxis.set_minor_formatter(plt.NullFormatter())  # optional: hide minor ticks
#
## Optional: adjust major ticks if you want them at specific positions
#log_positions = list(reversed(np.geomspace(eps_to_sweep[0], eps_to_sweep[-1], num=5)))
#ax.set_yticks(log_positions)
#ax.set_yticklabels([f'{val:.5f}' for val in log_positions])
#
## Colorbar with linear ks values
#cbar = fig.colorbar(im, ax=ax)
#cbar.set_label('ks value')
#cbar.formatter = FuncFormatter(lambda val, pos: f'{int(round(val))}')
#cbar.update_ticks()
#
#ax.set_xlabel('min-pts')
#ax.set_ylabel('eps')

plt.savefig('dbscan-varied-eps-minpts.png')
os.system('/usr/bin/xdg-open dbscan-varied-eps-minpts.png')
breakpoint()

