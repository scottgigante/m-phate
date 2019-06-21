import matplotlib
matplotlib.use("Agg")  # noqa
import numpy as np
import phate
import os
import graphtools
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scprep
from multiscalegraph.kernel import multiscale_kernel


data = loadmat("data/generalization/mnist_classifier_vanilla.mat")

trace = data['trace']

n_skip = 0
trace = trace[n_skip:]

n = trace.shape[0]
m = trace.shape[1]


trace = trace - np.mean(trace, axis=2)[:, :, None]
trace = trace / np.std(trace, axis=2)[:, :, None]

neuron_ids = np.tile(np.arange(m), n)
layer_ids = np.tile(data['layer'], n)
epoch = np.repeat(np.arange(n) + n_skip, m)

digit_ids = np.repeat(np.arange(10), 10)
digit_activity = np.array([np.sum(np.abs(trace[:, :, digit_ids == digit]), axis=2)
                           for digit in np.unique(digit_ids)])
most_active_digit = np.argmax(digit_activity, axis=0).flatten()


K = multiscale_kernel(trace, knn=2, decay=5,
                      interslice_knn=25)
graph = graphtools.Graph(
    K, precomputed="affinity", n_landmark=4000, n_svd=100)


###############
# Embed multislice graph
###############
phate_op = phate.PHATE(potential_method='sqrt')
m_phate = phate_op.fit_transform(graph)

plt.rc('font', size=14)
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3, figsize=(18, 6), sharex='all', sharey='all')
scprep.plot.scatter2d(m_phate, c=epoch, ax=ax1, ticks=False,
                      title='Epoch', label_prefix="M-PHATE")
scprep.plot.scatter2d(m_phate, c=layer_ids, ax=ax2, title='Layer',
                      ticks=False, label_prefix="M-PHATE")
scprep.plot.scatter2d(m_phate, c=most_active_digit, ax=ax3,
                      title='Most active digit',
                      ticks=False, label_prefix="M-PHATE")
plt.tight_layout()
plt.savefig("demonstration_phate.png")
