import matplotlib
matplotlib.use("Agg")  # noqa
import numpy as np
import phate
import os
import graphtools
import graphtools.utils
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scprep
from multiscalegraph.kernel import multiscale_kernel
from sklearn import linear_model
from sklearn.metrics import r2_score

data_dir = "data/generalization/"
n_skip = 0

out = {}
for filename in os.listdir(data_dir):
    data = loadmat(os.path.join(data_dir, filename))
    filename = filename.split('.')[0].split('_')
    filename = "_".join(filename[2:])
    trace = data['trace']
    loss = data['loss']
    val_loss = data['val_loss']
    trace = trace[n_skip:]
    n = trace.shape[0]
    m = trace.shape[1]
    trace = trace - np.mean(trace, axis=2)[:, :, None]
    trace = trace / np.std(trace, axis=2)[:, :, None]
    neuron_ids = np.tile(np.arange(m), n)
    layer_ids = np.tile(data['layer'], n)
    epoch = np.repeat(np.arange(n) + n_skip, m)
    digit_ids = np.repeat(np.arange(10), 10)
    digit_activity = np.array([np.sqrt(np.sum(trace[:, :, digit_ids == digit]**2, axis=2))
                               for digit in np.unique(digit_ids)])
    most_active_digit = np.argmax(digit_activity, axis=0).flatten()
    if filename in out:
        ph = out[filename]['phate']
    else:
        K = multiscale_kernel(trace, knn=2, decay=5,
                              fixed_bandwidth=False, interslice_knn=25, upweight=1)
        graph = graphtools.Graph(
            K, precomputed="affinity", n_landmark=3000, n_svd=20)
        phate_op = phate.PHATE(potential_method='sqrt')
        ph = phate_op.fit_transform(graph)
    out[filename] = {'phate': ph, 'epoch': epoch, 'most_active_digit': most_active_digit,
                     'layer_ids': layer_ids, 'loss': loss, 'val_loss': val_loss, 'digit_activity': digit_activity}

plt.rc('font', size=14)
fig, axes = plt.subplots(2, int(np.ceil(len(out) / 2)),
                         figsize=(4 * len(out) // 2, 8))
filenames = ['dropout', 'kernel_l1', 'kernel_l2',
             'vanilla', 'activity_l1', 'activity_l2', 'scrambled']
for ax, filename in zip(axes.flatten(), filenames):
    data = out[filename]
    title = " ".join([s.capitalize() for s in filename.split("_")]
                     if '_' in filename else [filename.capitalize()])
    scprep.plot.scatter2d(data['phate'], ax=ax,
                          xlabel='PHATE1' if filename in filenames[
                              4:] else None,
                          ylabel='PHATE2' if filename in [
                              filenames[0], filenames[4]] else None,
                          ticks=False,
                          c=data['most_active_digit'],
                          title=title, legend=False)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(xmin + (xmax - xmin) * 0.5 / 10, ymin + (ymax - ymin) * 9.25 / 10,
            'Train loss: {:.2f}'.format(data['loss'][-1, -1]),
            {'size': 12})
    ax.text(xmin + (xmax - xmin) * 0.5 / 10, ymin + (ymax - ymin) * 8.5 / 10,
            'Val loss: {:.2f}'.format(data['val_loss'][-1, -1]),
            {'size': 12})

scprep.plot.tools.generate_legend(
    cmap={i: plt.cm.tab10.colors[i] for i in range(10)},
    ax=axes[-1, -1], title='Most active digit',
    loc='center', fontsize=12)
axes[-1, -1].set_axis_off()
plt.tight_layout()
plt.savefig("generalization_phate.png")

import pandas as pd


def calculate_entropy(X, bins=10):
    hist = np.histogram2d(X[:, 0], X[:, 1], bins=bins)[0]
    p = hist / np.sum(hist)
    return np.sum(-p * np.where(p > 0, np.log2(p), 0))

performance = {}
entropy = {}
for filename in ['dropout', 'kernel_l1', 'kernel_l2', 'vanilla', 'activity_l1', 'activity_l2', 'scrambled']:
    data = out[filename]
    title = " ".join([s.capitalize() for s in filename.split("_")]
                     if '_' in filename else [filename.capitalize()])
    performance[title] = np.round(
        data['val_loss'][0, -1] - data['loss'][0, -1], 2)
    entropy[title] = np.round(calculate_entropy(data['phate']), 2)

performance_df = pd.DataFrame(columns=performance.keys())
performance_df.loc['Memorization error'] = performance
performance_df.loc['Visualization entropy'] = entropy
performance_df.to_latex()


regr = linear_model.LinearRegression()
x = performance_df.loc['Visualization entropy'].values[:, None]
y = performance_df.loc['Memorization error'].values[:, None]
regr.fit(x, y)
y_pred = regr.predict(x)
r2_score(y, y_pred)
