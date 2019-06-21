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

data_dir = os.path.expanduser(
    "data/task_switch/")
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
        elif False:
            ph = None
        else:
            K = multiscale_kernel(trace, knn=2, decay=5,
                                  fixed_bandwidth=False,
                                  interslice_knn=12, upweight=1)
            graph = graphtools.Graph(
                K, precomputed="affinity", n_landmark=3000,
                n_svd=20, n_jobs=20)
            phate_op = phate.PHATE(potential_method='sqrt', n_jobs=20)
            ph = phate_op.fit_transform(graph)
        out[filename] = {'phate': ph, 'epoch': epoch,
                         'most_active_digit': most_active_digit,
                         'neuron_ids': neuron_ids,
                         'layer_ids': layer_ids, 'loss': loss,
                         'val_loss': val_loss, 'val_accuracy': val_acc,
                         'task': np.repeat(data['task'], m),
                         'digit_activity': digit_activity}
    except Exception as e:
        print(filename, e)
        pass

n = len(np.unique(data['task']))
cmap = np.repeat(np.arange(n), 2)
cmap = cmap * 2 + (np.arange(len(cmap)) + 1) % 2
cmap = scprep.plot.tools.create_colormap(np.array(plt.cm.tab20.colors)[cmap])
linspace = np.linspace(0, 1 / (n * 2 - 1), 40)
cmap = matplotlib.colors.ListedColormap(cmap(np.concatenate([
    linspace + 2 * i / (n * 2 - 1) for i in range(n)])))

fig, ax = plt.subplots()
scprep.plot.tools.generate_colorbar(cmap=cmap, ax=ax)
plt.savefig("task_switch_colorbar.png")

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
        plot_idx = data['neuron_ids'] % 4 == 0
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
plt.savefig("task_switch_phate.png")

plt.rc('font', size=14)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
# ax.set_xscale('log')
# ax.set_yscale('log')
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
    ax.set_xticks([0, 4, 8, 12, 16])
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
plt.savefig("task_switch_phate_layer1.png")


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
plt.savefig("task_switch_phate_layer2.png")
