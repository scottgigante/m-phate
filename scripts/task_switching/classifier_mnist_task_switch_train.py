import numpy as np
import tensorflow as tf

import m_phate
import keras
import argparse

from scipy.io import savemat


keras.backend.set_session(tf.Session(config=m_phate.train.build_config()))


def mask_logits_to_softmax(y_true, y_pred):
    y_true_reshape = tf.reshape(y_true, (-1, 5, 2))
    y_true_task = y_true_reshape + y_true_reshape[:, :, ::-1]
    y_true_mask = tf.reshape(y_true_task, (-1, 10))
    logits_exp = tf.exp(y_pred)
    logits_exp_mask = y_true_mask * logits_exp
    y_pred = tf.divide(logits_exp_mask, tf.reduce_sum(
        logits_exp_mask, axis=1)[:, tf.newaxis])
    return y_pred


def masked_accuracy(y_true, y_pred):
    y_pred = mask_logits_to_softmax(y_true, y_pred)
    return keras.metrics.categorical_accuracy(y_true, y_pred)


def masked_crossentropy(y_true, y_pred):
    y_pred = mask_logits_to_softmax(y_true, y_pred)
    return keras.losses.categorical_crossentropy(y_true, y_pred)


def generator(X, Y, batch_size):
    while True:
        idx = np.random.choice(X.shape[0], X.shape[0], replace=False)
        for batch_idx in range(0, X.shape[0], batch_size):
            yield (X[idx][batch_idx:batch_idx + batch_size],
                   Y[idx][batch_idx:batch_idx + batch_size])


def rehearsal_generator(X, Y, X_rehearsal, Y_rehearsal, batch_size):
    gen = generator(X, Y, batch_size // 2)
    r_gen = generator(X_rehearsal, Y_rehearsal, batch_size // 2)
    while True:
        X_batch, Y_batch = next(gen)
        X_batch_r, Y_batch_r = next(r_gen)
        yield (np.vstack([X_batch, X_batch_r]),
               np.vstack([Y_batch, Y_batch_r]))


parser = argparse.ArgumentParser()
parser.add_argument('scheme', choices=['task', 'domain', 'class'], type=str)
parser.add_argument('--optimizer', '-o', choices=['adam', 'adagrad'], type=str)
parser.add_argument('--rehearsal', '-r', type=int, default=0)
parser.add_argument('--batch-size', '-b', type=int, default=128)

args = parser.parse_args()

rehearsal = args.rehearsal > 0

n_rehearsal = args.rehearsal
batch_size = args.batch_size

output_activation = None if args.scheme == 'task' else 'softmax'
hidden_activation = keras.layers.ReLU()
loss = masked_crossentropy if args.scheme == 'task' else 'categorical_crossentropy'
accuracy = masked_accuracy if args.scheme == 'task' else 'categorical_accuracy'
metric_name = 'masked' if args.scheme == 'task' else 'categorical'
output_shape = 2 if args.scheme == 'domain' else 10
if args.optimizer == 'adagrad':
    optimizer = keras.optimizers.Adagrad(lr=1e-4)
else:
    optimizer = keras.optimizers.Adam(lr=1e-5)


x_train, x_test, y_train, y_test = m_phate.data.load_mnist()


np.random.seed(42)
trace_idx = []
for i in range(10):
    trace_idx.append(np.random.choice(np.argwhere(
        y_test[:, i] == 1).flatten(), 10, replace=False))

trace_idx = np.concatenate(trace_idx)
trace_data = x_test[trace_idx]


inputs = keras.layers.Input(
    shape=(x_train.shape[1],), dtype='float32', name='inputs')
h1 = keras.layers.Dense(400, name='h1')(inputs)
h2 = keras.layers.Dense(400, name='h2')(hidden_activation(h1))
outputs = keras.layers.Dense(
    output_shape, activation=output_activation, name='output_all')(hidden_activation(h2))


model_trace = keras.models.Model(inputs=inputs, outputs=[h1, h2])
model = keras.models.Model(inputs=inputs, outputs=outputs)


trace = m_phate.train.BatchTraceHistory(trace_data, model_trace)
history = keras.callbacks.History()

model.compile(optimizer=optimizer, loss=loss,
              metrics=[accuracy])

rehearsal_data = []
rehearsal_gen = None
task = []
loss = []
val_loss = []
accuracy = []
val_accuracy = []
col_idxs = [[2 * i, 2 * i + 1]
            for i in range(5)]
task_idxs = [y_train[:, cols].sum(axis=1) == 1
             for cols in col_idxs]

if output_shape == 2:
    y_train = np.hstack([np.sum(y_train[:, np.arange(0, 10, 2)], axis=1)[:, None],
                         np.sum(y_train[:, np.arange(1, 11, 2)], axis=1)[:, None]])
    y_test = np.hstack([np.sum(y_test[:, np.arange(0, 10, 2)], axis=1)[:, None],
                        np.sum(y_test[:, np.arange(1, 11, 2)], axis=1)[:, None]])


for i, (train_idx, cols) in enumerate(zip(task_idxs, col_idxs)):
    if rehearsal and i > 0:
        new_rehearsal_data = np.random.choice(
            np.arange(x_train.shape[0])[task_idxs[0]],
            n_rehearsal // i, replace=False)
        rehearsal_data = [r[np.linspace(0, len(r), n_rehearsal // i,
                                        endpoint=False).astype(int)]
                          for r in rehearsal_data] + [new_rehearsal_data]
        task_gen = rehearsal_generator(x_train[train_idx], y_train[train_idx],
                                       x_train[np.concatenate(rehearsal_data)],
                                       y_train[np.concatenate(rehearsal_data)],
                                       batch_size)
    else:
        task_gen = generator(x_train[train_idx], y_train[train_idx],
                             batch_size)
    model.fit_generator(
        task_gen,
        steps_per_epoch=(np.sum(train_idx) // batch_size *
                         (1 if i == 0 and rehearsal else 2)),
        epochs=4,
        verbose=0, callbacks=[trace, history],
        validation_data=(x_test,
                         y_test))
    loss.append(history.history['loss'])
    val_loss.append(history.history['val_loss'])
    accuracy.append(history.history['{}_accuracy'.format(metric_name)])
    val_accuracy.append(history.history['val_{}_accuracy'.format(metric_name)])
    task.append(np.full_like(history.history['loss'], i, dtype=int))


filename = [args.scheme, args.optimizer]
if rehearsal:
    filename.append('rehearsal')
filename = "_".join(filename)

savemat(
    "data/task_switch/mnist_classifier_incremental_{}.mat".format(filename), {
        'trace': trace.trace, 'digit': y_test.argmax(1)[trace_idx],
        'layer': np.concatenate([np.repeat(i, int(layer.shape[1]))
                                 for i, layer in enumerate(model_trace.outputs)]),
        'loss': np.concatenate(loss),
        'val_loss': np.concatenate(val_loss),
        'accuracy': np.concatenate(accuracy),
        'val_accuracy': np.concatenate(val_accuracy),
        'task': np.concatenate(task)
    })
