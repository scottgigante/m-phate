import m_phate.train
import os
import numpy as np
import keras
from parameterized import parameterized

def test_config():
    config = m_phate.train.build_config(limit_gpu_fraction=0)
    assert config.device_count['GPU'] == 0
    config = m_phate.train.build_config(limit_gpu_fraction=0.5)
    assert config.gpu_options.per_process_gpu_memory_fraction == 0.5
    assert config.gpu_options.allow_growth
    config = m_phate.train.build_config(limit_cpu_fraction=2)
    assert config.intra_op_parallelism_threads == 2
    assert config.inter_op_parallelism_threads == 2
    config = m_phate.train.build_config(limit_cpu_fraction=0.5)
    assert config.intra_op_parallelism_threads == os.cpu_count() // 2
    assert config.inter_op_parallelism_threads == os.cpu_count() // 2
    config = m_phate.train.build_config(limit_cpu_fraction=-2)
    assert config.intra_op_parallelism_threads == os.cpu_count() - 1, (config.intra_op_parallelism_threads, os.cpu_count() - 1)
    assert config.inter_op_parallelism_threads == os.cpu_count() - 1
    config = m_phate.train.build_config(limit_cpu_fraction=0)
    assert config.intra_op_parallelism_threads == 1
    assert config.inter_op_parallelism_threads == 1

def test_trace_1_layer():
    # create data
    n_points = 100
    n_batch = 10
    n_trace = 10
    n_dim = 50
    n_hidden = 20
    x = np.random.normal(0, 1, [n_points, n_dim])

    # select trace examples
    trace_idx = np.arange(n_trace)
    trace_data = x[trace_idx]

    # build neural network
    inputs = keras.layers.Input(
        shape=(n_dim,), dtype='float32', name='inputs')
    h1 = keras.layers.Dense(n_hidden, name='h1')(inputs)
    outputs = keras.layers.Dense(n_dim, name='output_all')(h1)

    # build trace model helper
    model_trace = keras.models.Model(inputs=inputs, outputs=[h1])
    trace = m_phate.train.TraceHistory(trace_data, model_trace)
    batch_trace = m_phate.train.BatchTraceHistory(trace_data, model_trace)

    # compile network
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train network
    model.fit(x, x, batch_size=n_points // n_batch, epochs=1,
              verbose=0, callbacks=[trace, batch_trace])

    assert len(trace.trace) == 1
    assert trace.trace[0].shape == (n_hidden, n_trace)
    assert len(batch_trace.trace) == n_batch
    assert np.all([t.shape == (n_hidden, n_trace) for t in batch_trace.trace])

def test_trace_2_layer():
    # create data
    n_points = 100
    n_batch = 10
    n_trace = 10
    n_dim = 50
    n_hidden = 20
    x = np.random.normal(0, 1, [n_points, n_dim])

    # select trace examples
    trace_idx = np.arange(n_trace)
    trace_data = x[trace_idx]

    # build neural network
    inputs = keras.layers.Input(
        shape=(n_dim,), dtype='float32', name='inputs')
    h1 = keras.layers.Dense(n_hidden, name='h1')(inputs)
    h2 = keras.layers.Dense(n_hidden, name='h2')(h1)
    outputs = keras.layers.Dense(n_dim, name='output_all')(h2)

    # build trace model helper
    model_trace = keras.models.Model(inputs=inputs, outputs=[h1, h2])
    trace = m_phate.train.TraceHistory(trace_data, model_trace)
    batch_trace = m_phate.train.BatchTraceHistory(trace_data, model_trace)

    # compile network
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # train network
    model.fit(x, x, batch_size=n_points // n_batch, epochs=1,
              verbose=0, callbacks=[trace, batch_trace])

    assert len(trace.trace) == 1
    assert trace.trace[0].shape == (2*n_hidden, n_trace)
    assert len(batch_trace.trace) == n_batch
    assert np.all([t.shape == (2*n_hidden, n_trace) for t in batch_trace.trace])