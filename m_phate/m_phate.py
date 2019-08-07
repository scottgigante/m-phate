import phate
import graphtools
import tasklogger
import numpy as np

from . import kernel, utils


class M_PHATE(phate.PHATE):
    """Multislice PHATE operator which performs dimensionality reduction.

    M-PHATE embeds tensors of changing data in two or three dimensions
    for the visualization of the evolution of a dynamical system over time
    as described in Gigante et al, 2019 [1]_ for the visualization of
    the evolution of neural networks throughout learning.

    Parameters
    ----------

    n_components : int, optional, default: 2
        number of dimensions in which the data will be embedded

    intraslice_knn : int, optional, default: 2
        number of nearest neighbors on which to build intraslice kernels

    interslice_knn : int, optional, default: 25
        number of nearest neighbors on which to build interslice kernels

    decay : int, optional, default: 5
        sets decay rate of kernel tails.
        If None, alpha decaying kernel is not used

    n_landmark : int, optional, default: 4000
        number of landmarks to use in fast PHATE

    t : int, optional, default: 'auto'
        power to which the diffusion operator is powered.
        This sets the level of diffusion. If 'auto', t is selected
        according to the knee point in the Von Neumann Entropy of
        the diffusion operator

    gamma : float, optional, default: 0
        Informational distance constant between -1 and 1.
        `gamma=1` gives the PHATE log potential, `gamma=0` gives
        a square root potential.

    normalize : bool, optional, default: True
        If True, z-score the data

    n_pca : int, optional, default: 100
        Number of principal components to use for calculating
        neighborhoods. For extremely large datasets, using
        n_pca < 20 allows neighborhoods to be calculated in
        roughly log(n_samples) time.

    n_svd : int, optional, default: 100
        Number of singular vectors to use for calculating
        landmarks.

    n_jobs : integer, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
        n_jobs = -2, all CPUs but one are used

    random_state : integer or numpy.RandomState, optional, default: None
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global `numpy` random number generator

    verbose : `int` or `boolean`, optional (default: 1)
        If `True` or `> 0`, print status messages

    phate_kwargs : additional keyword arguments for phate.PHATE

    Attributes
    ----------

    X : array-like, shape=[n_samples, n_dimensions]

    embedding : array-like, shape=[n_samples, n_components]
        Stores the position of the dataset in the embedding space

    diff_op :  array-like, shape=[n_samples, n_samples] or [n_landmark, n_landmark]
        The diffusion operator built from the graph

    graph : graphtools.base.BaseGraph
        The graph built on the input data

    optimal_t : int
        The automatically selected t, when t = 'auto'.
        When t is given, optimal_t is None.

    Examples
    --------
    >>> import m_phate
    >>> m_phate_operator = m_phate.M_PHATE()
    >>> phate.plot.scatter2d(tree_phate, c=tree_clusters)

    References
    ----------
    .. [1] Gigante S, Charles AS, Krishnaswamy S and Mishne G (2019),
        *Visualizing the PHATE of Neural Networks*,
        `arXiv <http://arxiv.org/abs/>`_.
    """

    def __init__(self, n_components=2,
                 intraslice_knn=2, interslice_knn=25,
                 decay=5, t='auto', gamma=0, n_landmark=4000,
                 normalize=True, n_pca=100, n_svd=100,
                 n_jobs=-2, random_state=None, verbose=1,
                 knn=None,
                 **phate_kwargs):
        if knn is not None:
            warnings.warn("Argument `knn` is ambiguous and ignored. "
                          "Use `intraslice_knn` or `interslice_knn`.",
                          UserWarning)
        self.interslice_knn = interslice_knn
        self.n_svd = n_svd
        self.normalize = normalize
        return super().__init__(
            n_components=n_components,
            knn=intraslice_knn,
            decay=decay, t=t,
            n_pca=n_pca, gamma=gamma,
            n_landmark=n_landmark,
            n_jobs=n_jobs, random_state=random_state,
            verbose=verbose, **phate_kwargs)

    @property
    def intraslice_knn(self):
        return self.knn

    def fit(self, X):
        if not len(X.shape) == 3:
            raise ValueError("Expected X to be a tensor with three dimensions."
                             " Got shape {}".format(X.shape))

        if self.normalize:
            X = utils.normalize(X)

        tasklogger.log_start("multislice kernel")
        K = kernel.multislice_kernel(X,
                                     intraslice_knn=self.intraslice_knn,
                                     interslice_knn=self.interslice_knn,
                                     decay=self.decay,
                                     n_pca=self.n_pca,
                                     distance=self.knn_dist,
                                     n_jobs=self.n_jobs)
        tasklogger.log_complete("multislice kernel")
        tasklogger.log_start("graph and diffusion operator")
        n_landmark = self.n_landmark if self.n_landmark < K.shape[0] else None
        self.graph = graphtools.Graph(
            K,
            precomputed="affinity",
            n_landmark=n_landmark,
            n_svd=self.n_svd,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            **(self.kwargs))
        self.diff_op
        tasklogger.log_complete("graph and diffusion operator")
        result = super().fit(self.graph)
        return result

    def fit_transform(self, X, **kwargs):
        """Computes the diffusion operator and the position of the cells in the
        embedding space

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            input data with `n_samples` samples and `n_dimensions`
            dimensions. Accepted data types: `numpy.ndarray`,
            `scipy.sparse.spmatrix`, `pd.DataFrame`

        kwargs : further arguments for `M_PHATE.transform()`
            Keyword arguments as specified in :func:`~m_phate.M_PHATE.transform`

        Returns
        -------
        embedding : array, shape=[n_samples, n_dimensions]
            The cells embedded in a lower dimensional space using PHATE
        """
        tasklogger.log_start('M-PHATE')
        self.fit(X)
        embedding = self.transform(**kwargs)
        tasklogger.log_complete('M-PHATE')
        return embedding

    def _check_params(self):
        """Check M-PHATE parameters

        This allows us to fail early - otherwise certain unacceptable
        parameter choices, such as mds='mmds', would only fail after
        minutes of runtime.

        Raises
        ------
        ValueError : unacceptable choice of parameters
        """
        phate.utils.check_int(interslice_knn=self.interslice_knn,
                              n_svd=self.n_svd)
        phate.utils.check_positive(interslice_knn=self.interslice_knn,
                                   n_svd=self.n_svd)
        return super()._check_params()

    def set_params(self, **params):
        """Set the parameters on this estimator.

        Any parameters not given as named arguments will be left at their
        current value.

        Parameters
        ----------

        n_components : int, optional, default: 2
            number of dimensions in which the data will be embedded

        intraslice_knn : int, optional, default: 2
            number of nearest neighbors on which to build intraslice kernels

        interslice_knn : int, optional, default: 25
            number of nearest neighbors on which to build interslice kernels

        decay : int, optional, default: 5
            sets decay rate of kernel tails.
            If None, alpha decaying kernel is not used

        n_landmark : int, optional, default: 4000
            number of landmarks to use in fast PHATE

        t : int, optional, default: 'auto'
            power to which the diffusion operator is powered.
            This sets the level of diffusion. If 'auto', t is selected
            according to the knee point in the Von Neumann Entropy of
            the diffusion operator

        gamma : float, optional, default: 0
            Informational distance constant between -1 and 1.
            `gamma=1` gives the PHATE log potential, `gamma=0` gives
            a square root potential.

        n_pca : int, optional, default: 100
            Number of principal components to use for calculating
            neighborhoods. For extremely large datasets, using
            n_pca < 20 allows neighborhoods to be calculated in
            roughly log(n_samples) time.

        n_svd : int, optional, default: 100
            Number of singular vectors to use for calculating
            landmarks.

        n_jobs : integer, optional, default: 1
            The number of jobs to use for the computation.
            If -1 all CPUs are used. If 1 is given, no parallel computing code is
            used at all, which is useful for debugging.
            For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
            n_jobs = -2, all CPUs but one are used

        random_state : integer or numpy.RandomState, optional, default: None
            The generator used to initialize SMACOF (metric, nonmetric) MDS
            If an integer is given, it fixes the seed
            Defaults to the global `numpy` random number generator

        verbose : `int` or `boolean`, optional (default: 1)
            If `True` or `> 0`, print status messages
        """
        reset_kernel = False

        if 'interslice_knn' in params and params['interslice_knn'] != self.interslice_knn:
            self.interslice_knn = params['interslice_knn']
            reset_kernel = True
            del params['interslice_knn']

        if 'intraslice_knn' in params:
            params['knn'] = params['intraslice_knn']
            del params['intraslice_knn']

        if 'n_svd' in params and params['n_svd'] != self.n_svd:
            self._set_graph_params(n_svd=params['n_svd'])
            self.n_svd = params['n_svd']
            del params['n_svd']

        if reset_kernel:
            # can't reset the graph kernel without making a new graph
            self._reset_graph()

        return super().set_params(**params)
