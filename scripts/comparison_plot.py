import matplotlib
matplotlib.use("Agg")  # noqa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import m_phate
import phate
import scprep
import tasklogger
import os
import sys

from scipy.io import loadmat
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap, TSNE
from sklearn.neighbors import NearestNeighbors

try:
    data_dir = os.path.expanduser(sys.argv[1])
except IndexError:
    data_dir = "./data"

try:
    dataset = sys.argv[2]
except IndexError:
    dataset = "mnist"

data = loadmat(os.path.join(
    data_dir, "generalization/{}_classifier_vanilla.mat".format(dataset)))

trace = data['trace']
loss = data['val_loss']

n = trace.shape[0]
m = trace.shape[1]

neuron_ids = np.tile(np.arange(m), n)
layer_ids = np.tile(data['layer'], n)
epoch = np.repeat(np.arange(n), m)

digit_ids = np.repeat(np.arange(10), 10)
digit_activity = np.array([np.sum(np.abs(trace[:, :, digit_ids == digit]), axis=2)
                           for digit in np.unique(digit_ids)])
most_active_digit = np.argmax(digit_activity, axis=0).flatten()


tasklogger.log_start("Naive DR")
trace_flat = trace.reshape(-1, trace.shape[-1])
tasklogger.log_start("PHATE")
phate_naive_op = phate.PHATE(verbose=0)
phate_naive = phate_naive_op.fit_transform(trace_flat)
tasklogger.log_complete("PHATE")
tasklogger.log_start("DM")
dm_naive = m_phate.kernel.DM(phate_naive_op.graph)
tasklogger.log_complete("DM")
tasklogger.log_start("t-SNE")
tsne_naive = TSNE().fit_transform(trace_flat)
tasklogger.log_complete("t-SNE")
tasklogger.log_start("ISOMAP")
isomap_naive = Isomap().fit_transform(trace_flat)
tasklogger.log_complete("ISOMAP")
tasklogger.log_complete("Naive DR")

tasklogger.log_start("Multislice DR")
tasklogger.log_start("M-PHATE")
m_phate_op = m_phate.M_PHATE(verbose=0)
m_phate_data = m_phate_op.fit_transform(trace)
tasklogger.log_complete("M-PHATE")
tasklogger.log_start("DM")
dm_ms = m_phate.kernel.DM(m_phate_op.graph)
tasklogger.log_complete("DM")

geodesic_file = os.path.expanduser(
    "data/classifier_{}_geodesic.npy".format(dataset))
if False:
    tasklogger.log_start("geodesic distances")
    tasklogger.log_warning(
        "Warning: geodesic distance calculation will take a long time.")
    D_geo = m_phate_op.graph.shortest_path(distance='affinity')
    tasklogger.log_complete("geodesic distances")
    np.save(geodesic_file, D_geo)
else:
    D_geo = np.load(geodesic_file)

D_geo[~np.isfinite(D_geo)] = np.max(D_geo[np.isfinite(D_geo)])

tasklogger.log_start("ISOMAP")
isomap_ms = KernelPCA(2, kernel="precomputed").fit_transform(-0.5 * D_geo**2)
tasklogger.log_complete("ISOMAP")
tasklogger.log_start("t-SNE")
tsne_ms = TSNE(metric='precomputed').fit_transform(D_geo)
tasklogger.log_complete("t-SNE")
tasklogger.log_complete("Multislice DR")


plt.rc('font', size=14)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
scprep.plot.scatter2d(phate_naive, label_prefix="PHATE", ticks=False,
                      c=epoch, ax=ax1, legend=False)
scprep.plot.scatter2d(dm_naive, label_prefix="DM", ticks=False,
                      c=epoch, ax=ax2, legend=False)
scprep.plot.scatter2d(isomap_naive, label_prefix="Isomap", ticks=False,
                      c=epoch, ax=ax3, legend=False)
scprep.plot.scatter2d(tsne_naive, label_prefix="t-SNE", ticks=False,
                      c=epoch, ax=ax4, legend=False)
plt.tight_layout()
plt.savefig("{}_comparison_naive.png".format(dataset))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
scprep.plot.scatter2d(m_phate_data, label_prefix="PHATE", ticks=False,
                      c=epoch, ax=ax1, legend=False)
scprep.plot.scatter2d(dm_ms, label_prefix="DM", ticks=False,
                      c=epoch, ax=ax2, legend=False)
scprep.plot.scatter2d(isomap_ms, label_prefix="Isomap", ticks=False,
                      c=epoch, ax=ax3, legend=False)
scprep.plot.scatter2d(tsne_ms, label_prefix="t-SNE", ticks=False,
                      c=epoch, ax=ax4, legend=False)
plt.tight_layout()
plt.savefig("{}_comparison_multiscale.png".format(dataset))


def evaluate_loss(Y):
    loss_change = loss[0, :-1] - loss[0, 1:]
    Y_change = np.sum(
        (Y.reshape(n, m, -1)[:-1] - Y.reshape(n, m, -1)[1:])**2, axis=2) / Y.shape[-1]
    return scprep.stats.pairwise_correlation(Y_change, loss_change).mean()


def evaluate_within_slice(Y, k=40):
    neighbors_op = NearestNeighbors(k)
    result = []
    for e in np.unique(epoch):
        neighbors_op.fit(Y[epoch == e])
        _, Y_indices = neighbors_op.kneighbors()
        neighbors_op.fit(trace[e])
        _, trace_indices = neighbors_op.kneighbors()
        result.append([np.mean(np.isin(x, y))
                       for x, y in zip(Y_indices, trace_indices)])
    return np.mean(result)


def evaluate_between_slice(Y, k=40):
    neighbors_op = NearestNeighbors(k)
    result = []
    for neuron in np.unique(neuron_ids):
        neighbors_op.fit(Y[neuron_ids == neuron])
        _, Y_indices = neighbors_op.kneighbors()
        neighbors_op.fit(trace[:, neuron])
        _, trace_indices = neighbors_op.kneighbors()
        result.append([np.mean(np.isin(x, y))
                       for x, y in zip(Y_indices, trace_indices)])
    return np.mean(result)


embeddings = {'phate_ms': m_phate, 'dm_ms': dm_ms, 'isomap_ms': isomap_ms, 'tsne_ms': tsne_ms,
              'phate': phate_naive, 'dm': dm_naive, 'isomap': isomap_naive, 'tsne': tsne_naive}

df10 = pd.DataFrame({name: [evaluate_within_slice(Y, k=10), evaluate_between_slice(Y, k=10)] for name, Y in embeddings.items()},
                    index=['intraslice', 'interslice'])
df40 = pd.DataFrame({name: [evaluate_within_slice(Y, k=40), evaluate_between_slice(Y, k=40)] for name, Y in embeddings.items()},
                    index=['intraslice', 'interslice'])

print("k=10")
print(df10)
print()
print("k=40")
print(df40)
