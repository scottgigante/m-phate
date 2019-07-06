import matplotlib
matplotlib.use("Agg")  # noqa

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import m_phate
import scprep
import os
import sys

from scipy.io import loadmat
from sklearn import linear_model
from sklearn.metrics import r2_score

try:
    data_dir = os.path.expanduser(sys.argv[1])
except KeyError:
    data_dir = "./data"

data_dir = os.path.join(data_dir, "generalization")

out = {}
for filename in os.listdir(data_dir):
    data = loadmat(os.path.join(data_dir, filename))
    filename = filename.split('.')[0].split('_')
    filename = "_".join(filename[2:])
    trace = data['trace']
    loss = data['loss']
    val_loss = data['val_loss']

    n = trace.shape[0]
    m = trace.shape[1]
    neuron_ids = np.tile(np.arange(m), n)
    layer_ids = np.tile(data['layer'], n)
    epoch = np.repeat(np.arange(n), m)
    digit_ids = np.repeat(np.arange(10), 10)
    digit_activity = np.array([np.sqrt(np.sum(trace[:, :, digit_ids == digit]**2, axis=2))
                               for digit in np.unique(digit_ids)])
    most_active_digit = np.argmax(digit_activity, axis=0).flatten()

    if filename in out:
        m_phate_data = out[filename]['phate']
    else:
        m_phate_op = m_phate.M_PHATE()
        m_phate_data = m_phate_op.fit_transform(trace)

    out[filename] = {'phate': m_phate_data, 'epoch': epoch,
                     'most_active_digit': most_active_digit,
                     'layer_ids': layer_ids, 'loss': loss,
                     'val_loss': val_loss, 'digit_activity': digit_activity}

plt.rc('font', size=14)
filenames = ['dropout', 'kernel_l1', 'kernel_l2',
             'vanilla', 'activity_l1', 'activity_l2', 'scrambled']
nrow = 2
ncol = int(np.ceil(len(filenames) / 2))
fig, axes = plt.subplots(nrow, ncol,
                         figsize=(4 * ncol, 4 * nrow))
for i, ax, filename in zip(np.arange(len(filenames)), axes.flatten(), filenames):
    data = out[filename]
    title = " ".join([s.capitalize() for s in filename.split("_")]
                     if '_' in filename else [filename.capitalize()])
    print_ylabel = i % ncol == 0
    print_xlabel = i // ncol == nrow - 1
    scprep.plot.scatter2d(data['phate'], ax=ax,
                          xlabel='M-PHATE1' if print_xlabel else None,
                          ylabel='M-PHATE2' if print_ylabel else None,
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
plt.savefig("generalization.png")


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
print(performance_df)


regr = linear_model.LinearRegression()
x = performance_df.loc['Visualization entropy'].values[:, None]
y = performance_df.loc['Memorization error'].values[:, None]
regr.fit(x, y)
y_pred = regr.predict(x)
print("R^2:", r2_score(y, y_pred))
