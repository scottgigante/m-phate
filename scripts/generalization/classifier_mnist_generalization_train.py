import numpy as np
import tensorflow as tf

import keras
import m_phate
import argparse

from scipy.io import savemat


def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(
            "{} not in range [0.0, 1.0]".format(x))
    return x


parser = argparse.ArgumentParser()
parser.add_argument(
    '--regularizer', '-l', choices=['l1', 'l2'], default=None, type=str)
parser.add_argument('--regularize', '-r',
                    choices=['kernel', 'activity'], default=None, type=str)
parser.add_argument('--dropout', '-d', type=restricted_float, default=None)
parser.add_argument('--scrambled', '-s', action='set_true', default=False)
parser.add_argument('--batch-size', '-b', type=int, default=256)
parser.add_argument('--epochs', '-e', type=int, default=300)

args = parser.parse_args()

if (args.regularizer is not None and args.regualrize is None) or \
        (args.regularizer is None and args.regularize is not None):
    parser.error('--regularizer and --regularize must be given together')

if args.regularize is not None:
    reg_kwarg = 'activity_regularizer' if args.regularize == 'activity' else 'kernel_regularizer'
    reg_fn = keras.regularizers.l1 if args.regularizer == 'l1' else keras.regularizers.l2
    weight = keras.regularizers.l1 if args.regularizer == 'l1' else keras.regularizers.l2
    regularization = {reg_kwarg: reg_fn(1e-4)}
else:
    regularization = {}

keras.backend.set_session(tf.Session(config=m_phate.train.build_config()))

x_train, x_test, y_train, y_test = m_phate.data.load_mnist()

np.random.seed(42)
tf.set_random_seed(42)
trace_idx = []
for i in range(10):
    trace_idx.append(np.random.choice(np.argwhere(
        y_test[:, i] == 1).flatten(), 10, replace=False))

trace_idx = np.concatenate(trace_idx)
trace_data = x_test[trace_idx]

if args.scrambled:
    y_train = np.random.permutation(y_train)

lrelu = keras.layers.LeakyReLU(alpha=0.1)
if args.dropout is not None:
    dropout = keras.layers.Dropout(0.5)

inputs = keras.layers.Input(
    shape=(x_train.shape[1],), dtype='float32', name='inputs')
h1 = keras.layers.Dense(128,
                        **regularization,
                        name='h1')(inputs)
h1_out = lrelu(h1)
if args.dropout is not None:
    h1_out = dropout(h1_out)

h2 = keras.layers.Dense(128,
                        **regularization,
                        name='h2')(h1_out)
h2_out = lrelu(h2)
if args.dropout is not None:
    h2_out = dropout(h2_out)

h3 = keras.layers.Dense(128,
                        **regularization,
                        name='h3')(h2_out)
h3_out = lrelu(h3)
if args.dropout is not None:
    h3_out = dropout(h3_out)
outputs = keras.layers.Dense(
    10, activation='softmax', name='output_all')(h3_out)

model_trace = keras.models.Model(inputs=inputs, outputs=[h1, h2, h3])

trace = m_phate.train.TraceHistory(trace_data, model_trace)
history = keras.callbacks.History()

optimizer = keras.optimizers.Adam(lr=1e-5)
model = keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['categorical_accuracy', 'categorical_crossentropy'])

model.fit(x_train, y_train,
          batch_size=args.batch_size, epochs=args.epochs,
          verbose=0, callbacks=[trace, history],
          validation_data=(x_test,
                           y_test))


filename = []
if args.regularize is not None:
    filename.extend([args.regularize, args.regularizer])
if args.dropout is not None:
    filename.append("dropout")
if args.scrambled is not None:
    filename.append("scrambled")
if len(filename) == 0:
    filename.append("vanilla")
filename = "_".join(filename)

savemat(
    "data/generalization/mnist_classifier_{}.mat".format(filename), {
        'trace': trace.trace, 'digit': y_test.argmax(1)[trace_idx],
        'layer': np.concatenate([np.repeat(i, int(layer.shape[1]))
                                 for i, layer in enumerate(model_trace.outputs)]),
        'loss': history.history['categorical_crossentropy'],
        'val_loss': history.history['val_categorical_crossentropy'],
        'accuracy': history.history['categorical_accuracy'],
        'val_accuracy': history.history['val_categorical_accuracy'],
    })
