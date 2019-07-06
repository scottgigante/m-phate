===========================================================================
M-PHATE
===========================================================================

.. raw:: html

    <a href="https://pypi.org/project/m-phate/"><img src="https://img.shields.io/pypi/v/m-phate.svg" alt="Latest PyPi version"></a>

.. raw:: html

    <a href="https://travis-ci.com/scottgigante/m-phate"><img src="https://api.travis-ci.com/scottgigante/m-phate.svg?branch=master" alt="Travis CI Build"></a>

.. raw:: html

    <a href="https://m-phate.readthedocs.io/"><img src="https://img.shields.io/readthedocs/m-phate.svg" alt="Read the Docs"></img></a>

.. raw:: html

    <a href="https://coveralls.io/github/scottgigante/m-phate?branch=master"><img src="https://coveralls.io/repos/github/scottgigante/m-phate/badge.svg?branch=master" alt="Coverage Status"></img></a>

.. raw:: html

    <a href="https://twitter.com/scottgigante"><img src="https://img.shields.io/twitter/follow/scottgigante.svg?style=social&label=Follow" alt="Twitter"></a>

.. raw:: html

    <a href="https://github.com/scottgigante/m-phate/"><img src="https://img.shields.io/github/stars/scottgigante/m-phate.svg?style=social&label=Stars" alt="GitHub stars"></a>

Multislice PHATE (M-PHATE) is a dimensionality reduction algorithm for the visualization of changing data. To learn more about M-PHATE, you can read our preprint on arXiv in which we apply it to the evolution of neural networks over the course of training.

.. toctree::
    :maxdepth: 2

    installation
    reference

Quick Start
===========

You can use `m-phate` as follows::

    import numpy as np
    import m_phate
    import scprep
    
    # create fake data
    n_time_steps = 100
    n_points = 50
    n_dim = 20
    np.random.seed(42)
    data = np.cumsum(np.random.normal(0, 1, (n_time_steps, n_points, n_dim)), axis=0)
    
    # embedding
    m_phate_op = m_phate.M_PHATE()
    m_phate_data = m_phate_op.fit_transform(data)
    
    # plot
    time = np.repeat(np.arange(n_time_steps), n_points)
    scprep.plot.scatter2d(m_phate_data, c=time)

Help
====

If you have any questions or require assistance using M-PHATE, please `file an issue <http://github.com/scottgigante/m-phate/issues>`_.