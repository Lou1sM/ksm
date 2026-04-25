import numpy as np
import warnings
warnings.filterwarnings('ignore')  # suppress all warnings
import itertools
from scipy.stats import qmc
import json
import matplotlib.pyplot as plt
import pandas as pd
import math
from utils import profile_lines, print_stats

def manual_bridson_sampling(k=30, d=0.1):
    # Calculate region size
    region_size = (4/math.pi) * ((k*d) + d*math.sqrt(2*k/math.sqrt(3))) / 2
    width, height = region_size, region_size

    # Setup acceleration grid
    cell_size = d / math.sqrt(2)
    grid = {}  # Changed to dictionary for sparse storage

    # Setup lists
    active = []
    points = []

    # Add first random point
    x, y = width * np.random.random(), height * np.random.random()
    active.append((x, y))
    points.append((x, y))
    grid_x, grid_y = int(x / cell_size), int(y / cell_size)
    grid[(grid_x, grid_y)] = [0]  # Store indices in list

    # Try to add more points
    while active and len(points) < k:
        idx = np.random.randint(len(active))
        p = active[idx]

        # Try to find a valid point around p
        found = False
        for _ in range(30):
            theta = 2 * math.pi * np.random.random()
            radius = d + np.random.random() * d
            new_x = p[0] + radius * math.cos(theta)
            new_y = p[1] + radius * math.sin(theta)

            # Check if point is too close to existing points
            grid_x, grid_y = int(new_x / cell_size), int(new_y / cell_size)
            valid = True

            # Check surrounding cells
            for i in range(grid_x - 2, grid_x + 3):
                for j in range(grid_y - 2, grid_y + 3):
                    if (i, j) in grid:
                        for point_idx in grid[(i, j)]:
                            px, py = points[point_idx]
                            dx, dy = new_x - px, new_y - py
                            if dx*dx + dy*dy < d*d:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break

            if valid:
                # Add new point
                points.append((new_x, new_y))
                point_idx = len(points) - 1
                active.append((new_x, new_y))

                # Add to grid
                if (grid_x, grid_y) not in grid:
                    grid[(grid_x, grid_y)] = []
                grid[(grid_x, grid_y)].append(point_idx)

                found = True
                break

        if not found:
            del active[idx]

    points = np.array(points)
    return points, region_size

def bridson_sampling(k, dim, min_dist):
    # Calculate region size
    region_size = (4/math.pi) * ((k*min_dist) + min_dist*math.sqrt(2*k/math.sqrt(3))) / 2
    # Setup acceleration grid
    cell_size = min_dist / math.sqrt(dim)
    grid = {}

    # Setup lists
    active = []
    points = []

    # Add first random point
    first_point = tuple(region_size * np.random.random() for _ in range(dim))
    active.append(first_point)
    points.append(first_point)
    grid_coord = tuple(int(x / cell_size) for x in first_point)
    grid[grid_coord] = [0]

    # Try to add more points
    while active and len(points) < k:
        idx = np.random.randint(len(active))
        p = active[idx]

        # Try to find a valid point around p
        found = False
        for _ in range(30):
            # Generate random point in annulus
            radius = min_dist + np.random.random() * min_dist
            direction = np.random.normal(size=dim)
            direction = direction / np.linalg.norm(direction)
            new_point = tuple(p[i] + radius * direction[i] for i in range(dim))

            # Check if point is in bounds
            if any(x < 0 or x > region_size for x in new_point):
                continue

            # Check if point is too close to existing points
            grid_coord = tuple(int(x / cell_size) for x in new_point)
            valid = True

            # Check surrounding cells
            ranges = [range(g - 2, g + 3) for g in grid_coord]
            for coords in itertools.product(*ranges):
                if coords in grid:
                    for point_idx in grid[coords]:
                        dist_sq = sum((new_point[i] - points[point_idx][i])**2 for i in range(dim))
                        if dist_sq < min_dist**2:
                            valid = False
                            break
                if not valid:
                    break

            if valid:
                # Add new point
                points.append(new_point)
                point_idx = len(points) - 1
                active.append(new_point)

                # Add to grid
                if grid_coord not in grid:
                    grid[grid_coord] = []
                grid[grid_coord].append(point_idx)

                found = True
                break

        if not found:
            del active[idx]

    points = np.array(points)
    return points, region_size

def create_synthetic_data(n, k, dim, min_dist):
    assert 1 <= k <= n
    if dim==2:
        centroids, _ = manual_bridson_sampling(k, min_dist)
    centroids, _ = bridson_sampling(k, dim=dim, min_dist=min_dist)
    #centroids, _ = engine.sample(k)
    #engine = qmc.PoissonDisk(d=dim, radius=min_dist)
    #if k>1:
        #breakpoint()
    points_per_c = math.ceil(n/k)
    tiled_centroids = np.repeat(centroids, points_per_c, axis=0)[:n]
    if VARY_STDS:
        stds = np.maximum(0, 1 + np.random.randn(n)) # sample stds normally distributed around 1
    else:
        stds = np.ones(n)
    residual_displacements = np.random.randn(n, dim) * np.expand_dims(stds, 1)
    X = tiled_centroids + residual_displacements
    y = np.repeat(np.arange(k), points_per_c)[:n]
    return X, y

if __name__ == '__main__':
    from kstar_means import KStarMeans, lablled_cluster_metrics
    from sklearn.cluster import DBSCAN, HDBSCAN, MeanShift, AffinityPropagation, OPTICS
    from xmeans import DummyXMeans
    from time import time
    from tqdm import tqdm
    from utils import profile_lines, print_stats
    from sweepkm import ElbowMethod
    import os

    methods = {}
    methods['ksm'] = KStarMeans(subinit='ksm', run_checks=False)
    #methods['dbscan'] = DBSCAN()
    #methods['hdbscan'] = HDBSCAN()
    #methods['elbow'] = ElbowMethod()
    #methods['xmeans'] = DummyXMeans(kmax=int(1000**0.5))
    #all_pred_names = ['pfc', 'dbscan', 'hdbscan', 'xmeans', 'elbow']
    #if ARGS.include_slow_baselines:
        #methods['affinity'] = AffinityPropagation()
        #methods['optics'] = OPTICS(min_samples=2)
        #all_pred_names += ['meanshift', 'affinity', 'optics']

    VARY_STDS = False
    results_dir = 'results/synth-varied-std' if VARY_STDS else 'results/synth'
    results_dir += ''.join(methods.keys())
    os.makedirs(results_dir, exist_ok=True)
    #nc_range = range(1, 50+1, 1)
    nc_range = [20]
    ns = 1
    all_summaries = {}
    #min_dist_range = (2, 3, 4, 5)
    min_dist_range = (2, 3, 5,)
    for sweep_dim in (2,):#, 3, 4, 5):
        print('dim', sweep_dim)
        for min_dist in min_dist_range:
            print('mindist', min_dist)
            columns = [x+suffix for x in methods.keys() for suffix in ('-nc', '-runtime', '-acc', '-ari', '-nmi')] + ['gt']
            results = pd.DataFrame(columns=columns)#, index=np.arange(len(nc_range)*ns))
            for k in tqdm(nc_range):
                for ns_idx in range(ns):
                    x_idx = (k-1)*ns + ns_idx
                    results.loc[x_idx, 'gt'] = k
                    X, y = create_synthetic_data(1000, k, dim=sweep_dim, min_dist=min_dist)
                    if k in [10, 20, 50] and (ns_idx==0):# and (min_dist==5):
                        plt.scatter(X[:,0], X[:,1], c=y, cmap='tab20')
                        plt.title(f'Synthetic Data with {k} Clusters')
                        png_fp = os.path.join(results_dir, f'min_dist{min_dist}-plot-{k}clusters.png')
                        plt.savefig(png_fp)
                        print(png_fp)
                        os.system(f'/usr/bin/xdg-open {png_fp}')
                        plt.clf()
                    continue
                    for pname, model in methods.items():
                        starttime = time()
                        preds = model.fit_predict(X)
                        preds[preds==-1] = preds.max()+1 # treat (h)dbscan's -1 as a class
                        runtime = time()-starttime
                        metrics = lablled_cluster_metrics(preds, y)
                        for mname, mscore in metrics.items():
                            results.loc[x_idx, f'{pname}-{mname}'] = 100*mscore
                        pred_nc = len(set(preds)) - (1 if -1 in preds else 0)
                        results.loc[x_idx, f'{pname}-nc'] = pred_nc
                        results.loc[x_idx, f'{pname}-runtime'] = runtime*2
                        results.loc[x_idx, f'{pname}-mse'] = (pred_nc - k)**2
                        results.loc[x_idx, f'{pname}-kacc'] = float(pred_nc == k)

            continue
            if not ( not results.isna().any().any()):
                breakpoint()
            print(results.mean(axis=0))
            results.to_csv(os.path.join(results_dir, f'dim{sweep_dim}-synth-mindist{min_dist}.csv'))
            summary_results = {}
            for metric_name in ['mse', 'kacc', 'acc', 'ari', 'nmi', 'runtime', 'nc']:
                summary_results.update({f'{k}-{metric_name}': results[f'{k}-{metric_name}'].mean() for k in methods.keys()})
            os.makedirs('results/pfc', exist_ok=True)
            print({k:f'{v:.4f}' for k,v in summary_results.items()})
            with open(os.path.join(results_dir, f'dim{sweep_dim}-synth-mindist{min_dist}.json'), 'w') as f:
                json.dump(summary_results, f, default=float)
            all_summaries[f'dim{sweep_dim}-mindist{min_dist}'] = summary_results
    #df = pd.DataFrame(all_summaries, index=[f'sythetic d={d}' for d in min_dist_range])
    df = pd.DataFrame(all_summaries).T
    breakpoint()
    df.to_latex(os.path.join(results_dir, 'summary_results_table.tex'))
    df.to_csv(os.path.join(results_dir, 'summary_results.csv'))
