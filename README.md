# M-PHATE

Multislice PHATE (M-PHATE) is a dimensionality reduction algorithm for the visualization of changing data. To learn more about M-PHATE, you can read our preprint on arXiv in which we apply it to the evolution of neural networks over the course of training.

## Installation

```
pip install --user git+https://github.com/scottgigante/m-phate.git
```

## Usage

```
import numpy as np
import m_phate
import scprep

# create fake data
n_time_steps = 100
n_points = 50
n_dim = 50
data = np.cumsum(np.random.normal(0, 1, (n_time_steps, n_points, n_dim)), axis=0)

# embedding
m_phate_op = m_phate.M_PHATE()
m_phate_data = m_phate_op.fit_transform(data)

# plot
time = np.repeat(np.arange(n_time_steps), n_points)
scprep.plot.scatter2d(m_phate_data, c=time)
```

## Network training

To apply M-PHATE to neural networks, we provide helper classes to store the samples from the network during training. In order to use these, you must install [`tensorflow`](https://www.tensorflow.org/install) and [`keras`](https://keras.io/#installation).

```
import keras
import m_phate.train

# provide minimum working example
```

## Parameter tuning

TODO.

## Figure reproduction

We provide scripts to reproduce all of the empirical figures in the preprint. 

To run them:

```
git clone https://github.com/scottgigante/m-phate
cd m-phate
pip install --user .
DATA_DIR=~/data/checkpoints/m_phate # change this if you want to store the data elsewhere

chmod +x scripts/generalization/generalization_train.sh
chmod +x scripts/task_switching/classifier_mnist_task_switch_train.sh

./scripts/generalization/generalization_train.sh $DATA_DIR
./scripts/task_switching/classifier_mnist_task_switch_train.sh $DATA_DIR

python scripts/demonstration_plot.py $DATA_DIR
python scripts/comparison_plot.py $DATA_DIR
python scripts/generalization_plot.py $DATA_DIR
python scripts/task_switch_plot.py $DATA_DIR
```

## TODO

* Provide support for PyTorch
* Notebook examples for:
  * Classification, tf
  * Autoencoder, tf
  * Classification, keras
  * Autoencoder, keras
  * Classification, pytorch
  * Autoencoder, pytorch
* Parameter tuning discussion notebook
* Submit to pypi
* Build readthedocs page
* shields.io badges

### Help

If you have any questions, please feel free to [open an issue](https://github.com/scottgigante/m-phate/issues).