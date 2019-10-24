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
import scipy.stats
from sklearn import cluster, metrics

from joblib import Parallel, delayed


try:
    data_dir = os.path.expanduser(sys.argv[1])
except IndexError:
    data_dir = "./data"

data_dir = os.path.join(data_dir, "task_switch")

n_skip = 50
n_step = 50

out = {}
for filename in os.listdir(data_dir):
    try:
        data = loadmat(os.path.join(data_dir, filename))
        filename = ".".join(filename.split('.')[:-1]).split('_')
        filename = "_".join(filename[2:])
        trace = data['trace']
        loss = data['loss']
        val_loss = data['val_loss']
        val_acc = data['val_accuracy']
        trace = trace[n_skip::n_step]

        n = trace.shape[0]
        m = trace.shape[1]
        neuron_ids = np.tile(np.arange(m), n)
        layer_ids = np.tile(data['layer'], n)
        epoch = np.repeat(np.arange(n) + n_skip, m)
        digit_ids = np.repeat(np.arange(10), 10)
        digit_activity = np.array([np.sqrt(np.sum(trace[:, :, digit_ids == digit]**2, axis=2))
                                   for digit in np.unique(digit_ids)])
        most_active_digit = np.argmax(digit_activity, axis=0).flatten()

        if filename in out:
            m_phate_data = out[filename]['phate']
        else:
            m_phate_op = m_phate.M_PHATE(interslice_knn=12)
            m_phate_data = m_phate_op.fit_transform(trace)

        out[filename] = {'phate': m_phate_data, 'epoch': epoch,
                         'most_active_digit': most_active_digit,
                         'neuron_ids': neuron_ids,
                         'layer_ids': layer_ids, 'loss': loss,
                         'val_loss': val_loss, 'val_accuracy': val_acc,
                         'task': np.repeat(data['task'][0,n_skip::n_step], m),
                         'digit_activity': digit_activity}
    except Exception as e:
        print(filename, e)
        pass

cmap = scprep.plot.colors.tab10_continuous(5)
fig, ax = plt.subplots()
scprep.plot.tools.generate_colorbar(cmap=cmap, ax=ax)
plt.savefig("task_switch_colorbar.png")

plt.rc('font', size=14)
colnames = ['task', 'domain', 'class']
rownames = ['adam_rehearsal', 'adagrad', 'adam']
fig, axes = plt.subplots(len(rownames), len(colnames),
                         figsize=(4 * len(colnames), 4 * len(rownames)))
for rowname, axis_row in zip(rownames, axes):
    for colname, ax in zip(colnames, axis_row):
        filename = "incremental_{}_{}".format(colname, rowname)
        data = out[filename]
        label = 'rehearsal' if 'rehearsal' in rowname else rowname
        plot_idx = data['neuron_ids'] % 4 == 0
        scprep.plot.scatter2d(
            data['phate'][plot_idx], ax=ax,
            xlabel='M-PHATE1' if rowname == rownames[-1] else None,
            ylabel='M-PHATE2' if colname == colnames[0] else None,
            ticks=False,
            c=data['epoch'][plot_idx],
            cmap=cmap,
            title="{}: {}".format(colname.capitalize(), label),
            legend=False)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(xmin + (xmax - xmin) * 5 / 10, ymin + (ymax - ymin) * 9.25 / 10,
                'Accuracy: {:.0f}%'.format(data['val_accuracy'][-1, -1] * 100),
                {'size': 12})
        ax.text(xmin + (xmax - xmin) * 5 / 10, ymin + (ymax - ymin) * 8.5 / 10,
                'Loss: {:.2f}'.format(data['val_loss'][-1, -1]),
                {'size': 12})

plt.tight_layout()
plt.savefig("task_switch.png")


plt.rc('font', size=14)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
colors = {'adagrad': plt.cm.tab20.colors[:2],
          'adam': plt.cm.tab20.colors[6:8],
          'adam_rehearsal': plt.cm.tab20.colors[2:4],
          'adagrad_rehearsal': plt.cm.tab20.colors[4:6]}
rownames = ['task', 'domain', 'class']
colnames = ['adam_rehearsal', 'adagrad', 'adam']
for rowname, ax in zip(rownames, axes):
    for colname in colnames:
        filename = "incremental_{}_{}".format(rowname, colname)
        data = out[filename]
        c_val, c_train = colors[colname]
        label = 'rehearsal' if 'rehearsal' in colname else colname
        ax.plot(data['val_loss'].flatten(), color=c_val,
                label=label + ' val')
        ax.plot(data['loss'].flatten(), color=c_train,
                linestyle='-.', label=label + ' train')
    ax.set_title('Incremental ' + rowname)
    ax.set_xticks(np.linspace(0, data['val_loss'].shape[1], 6))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Categorical cross entropy")

axes[2].legend(bbox_to_anchor=[1.02, 0.8])
plt.tight_layout()
plt.savefig("task_switch_loss.png")


plt.rc('font', size=14)
colnames = ['task', 'domain', 'class']
rownames = ['adam_rehearsal', 'adagrad', 'adam']
fig, axes = plt.subplots(len(rownames), len(colnames),
                         figsize=(4 * len(colnames), 4 * len(rownames)))
for rowname, row in zip(rownames, axes):
    for colname, ax in zip(colnames, row):
        filename = "incremental_{}_{}".format(colname, rowname)
        data = out[filename]
        label = 'rehearsal' if 'rehearsal' in rowname else rowname
        plot_idx = data['layer_ids'][0] == 0
        scprep.plot.scatter2d(
            data['phate'][plot_idx], ax=ax,
            xlabel='PHATE1' if rowname == rownames[-1] else None,
            ylabel='PHATE2' if colname == colnames[0] else None,
            ticks=False,
            c=data['epoch'][plot_idx],
            cmap=cmap,
            title="{}: {}".format(colname.capitalize(), label),
            legend=False)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(xmin + (xmax - xmin) * 5 / 10, ymin + (ymax - ymin) * 9.25 / 10,
                'Accuracy: {:.0f}%'.format(data['val_accuracy'][-1, -1] * 100),
                {'size': 12})
        ax.text(xmin + (xmax - xmin) * 5 / 10, ymin + (ymax - ymin) * 8.5 / 10,
                'Loss: {:.2f}'.format(data['val_loss'][-1, -1]),
                {'size': 12})

plt.tight_layout()
plt.savefig("task_switch_layer1.png")


plt.rc('font', size=14)
colnames = ['task', 'domain', 'class']
rownames = ['adam_rehearsal', 'adagrad', 'adam']
fig, axes = plt.subplots(len(rownames), len(colnames),
                         figsize=(4 * len(colnames), 4 * len(rownames)))
for rowname, row in zip(rownames, axes):
    for colname, ax in zip(colnames, row):
        filename = "incremental_{}_{}".format(colname, rowname)
        data = out[filename]
        label = 'rehearsal' if 'rehearsal' in rowname else rowname
        plot_idx = data['layer_ids'][0] == 1
        scprep.plot.scatter2d(
            data['phate'][plot_idx], ax=ax,
            xlabel='PHATE1' if rowname == rownames[-1] else None,
            ylabel='PHATE2' if colname == colnames[0] else None,
            ticks=False,
            c=data['epoch'][plot_idx],
            cmap=cmap,
            title="{}: {}".format(colname.capitalize(), label),
            legend=False)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(xmin + (xmax - xmin) * 5 / 10, ymin + (ymax - ymin) * 9.25 / 10,
                'Accuracy: {:.0f}%'.format(data['val_accuracy'][-1, -1] * 100),
                {'size': 12})
        ax.text(xmin + (xmax - xmin) * 5 / 10, ymin + (ymax - ymin) * 8.5 / 10,
                'Loss: {:.2f}'.format(data['val_loss'][-1, -1]),
                {'size': 12})

plt.tight_layout()
plt.savefig("task_switch_layer2.png")


####################
# Quantification
####################


def ari(x, y, k, seed=None):
    clusters_x = cluster.KMeans(k, random_state=seed).fit_predict(x)
    clusters_y = cluster.KMeans(k, random_state=seed).fit_predict(y)
    return metrics.adjusted_rand_score(clusters_x, clusters_y)


def ari_score(data, min_clusters=3, max_clusters=8, n_rep=20, n_jobs=20):
    phate = data['phate']
    epoch = data['epoch']
    task = data['task']
    scores = []
    with Parallel(n_jobs) as p:
        for i in range(np.max(task).astype(int)):
            pre_switch_epoch = np.max(epoch[task==i])
            post_switch_epoch = np.min(epoch[task==i+1])
            epoch_scores = []
            for k in range(min_clusters, max_clusters+1):
                epoch_scores.append(p(delayed(ari)(phate[epoch==pre_switch_epoch],
                                                   phate[epoch==post_switch_epoch+3],
                                                   k=k, seed=seed)
                                      for seed in range(n_rep)))
            scores.append(epoch_scores)
    return np.mean(scores)


df = pd.DataFrame(columns=['loss', 'ari'])
for colname in colnames:
    for rowname in rownames:
        filename = "incremental_{}_{}".format(colname, rowname)
        data = out[filename]
        df.loc[filename] = [data['val_loss'].flatten()[-1],
                            ari_score(data)]

print(df.round(3))
df.to_csv("task_switch_performance.csv")
print("rho = {}".format(scipy.stats.pearsonr(df['loss'], df['ari'])[0]))
