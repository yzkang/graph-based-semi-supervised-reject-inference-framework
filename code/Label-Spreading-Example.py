# -*- coding: utf-8 -*-

"""
@Time      : 2020-07-14 11:28
@Author    : Yanzhe Kang <kyz1994@tju.edu.cn>
@License   : GPL
@Desc      : Example of label spreading algorithm

-------------------------------------------------------------------------------
This demo illustrate that if data examples lie inside their own manifold,
their labels can spread correctly around the circle, which proves that label
spreading algorithms can spread specific labels to unlabeled data under
manifold assumption.
-------------------------------------------------------------------------------
References:

1. Zhou, D. , Bousquet, O. , Lal, T. N. , Weston, J. , & Olkopf, B. S. . (2003).
   Learning with local and global consistency. Advances in neural information
   processing systems, 16(3).

2. https://scikit-learn.org/stable/modules/label_propagation.html

"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.datasets import make_circles

from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y
from sklearn.exceptions import ConvergenceWarning


class BaseModel(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):

    """
    Base model class for label spreading algorithm

    Parameters
    ----------
    kernel : Kernel function. Only 'knn' strings are valid inputs.

    n_neighbors : Parameter for knn kernel

    alpha : Clamping factor; Clamping allows the algorithm to change the weight
            of the true ground labeled data to some degree

    max_iter : Parameter for iterations

    tol : Convergence tolerance: threshold to consider the system at steady
        state

    n_jobs : The number of parallel jobs to run.

    """

    def __init__(self, kernel='knn', n_neighbors=7, alpha=2, max_iter=30,
                 tol=1e-4, n_jobs=None):

        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.n_jobs = n_jobs
        self._variant = ''  # ensure that the algorithm is label spreading

    def _get_kernel(self, X, y=None):

        if self.kernel == "knn":
            if self.nn_fit is None:
                self.nn_fit = NearestNeighbors(self.n_neighbors,
                                               n_jobs=self.n_jobs).fit(X)
            if y is None:
                return self.nn_fit.kneighbors_graph(self.nn_fit._fit_X,
                                                    self.n_neighbors,
                                                    mode='connectivity')
            else:
                return self.nn_fit.kneighbors(y, return_distance=False)

        elif callable(self.kernel):
            if y is None:
                return self.kernel(X, X)
            else:
                return self.kernel(X, y)

        else:
            raise ValueError("%s is not a valid kernel. Only rbf and knn"
                             " or an explicit function "
                             " are supported at this time." % self.kernel)

    def _build_graph(self):
        raise NotImplementedError("Graph construction must be implemented"
                                  " to fit a label propagation model.")

    def fit(self, X, y):

        """
        learning the data structure
        :param X: input matrix of examples
        :param y: corresponding label matrix of examples
        :return: self
        """

        X, y = check_X_y(X, y)
        self.X_ = X
        check_classification_targets(y)

        # 1. actual graph construction
        graph_matrix = self._build_graph()

        # 2. how much classes in the input examples
        classes = np.unique(y)
        classes = (classes[classes != -1])
        self.classes_ = classes

        n_examples, n_classes = len(y), len(classes)

        # 3. Clamping factor
        alpha = self.alpha

        # exception handling
        if self._variant == 'spreading' and \
                (alpha is None or alpha <= 0.0 or alpha >= 1.0):
            raise ValueError('alpha=%s is invalid: it must be inside '
                             'the open interval (0, 1)' % alpha)

        y = np.asarray(y)  # labels of given examples

        # unlabeled = y == -1

        # 4. initialize label distributions of given examples
        self.label_distributions_ = np.zeros((n_examples, n_classes))
        for label in classes:
            self.label_distributions_[y == label, classes == label] = 1
        # label_distributions_ = [[1, 0], [0,0], ..., [0,0], [0,1]]

        y_static = np.copy(self.label_distributions_)

        # 5. change the weight of the true ground labeled data to alpha degree
        if self._variant == 'spreading':
            y_static *= 1 - alpha

        l_previous = np.zeros((self.X_.shape[0], n_classes))

        if sparse.isspmatrix(graph_matrix):
            graph_matrix = graph_matrix.tocsr()

        # 6. Yˆ(t+1) ← αLYˆ(t) + (1 − α)Yˆ(0)
        for self.n_iter_ in range(self.max_iter):

            if np.abs(self.label_distributions_ - l_previous).sum() < self.tol:
                break

            l_previous = self.label_distributions_

            self.label_distributions_ = safe_sparse_dot(
                graph_matrix, self.label_distributions_)

            if self._variant == 'spreading':
                # clamp
                self.label_distributions_ = np.multiply(
                    alpha, self.label_distributions_) + y_static

            self.n_iter_ += 1

        # 7. normalize the label distributions
        normalizer = np.sum(self.label_distributions_, axis=1)[:, np.newaxis]
        self.label_distributions_ /= normalizer

        # 8. set the transduction item
        transduction = self.classes_[np.argmax(self.label_distributions_,
                                               axis=1)]
        self.transduction_ = transduction.ravel()  # array to vector

        return self


class LabelSpreadingAlgorithm(BaseModel):

    """
    Class for label spreading algorithm
    """

    _variant = 'spreading'

    def __init__(self, kernel='knn', n_neighbors=7, alpha=0.2, max_iter=10,
                 tol=1e-3, n_jobs=None):

        # this one has different base parameters
        super().__init__(kernel=kernel,
                         n_neighbors=n_neighbors, alpha=alpha,
                         max_iter=max_iter, tol=tol, n_jobs=n_jobs)

    def _build_graph(self):

        """
        Computes the graph laplacian matrix for label spreading
        """

        # compute affinity matrix (or gram matrix)

        if self.kernel == 'knn':
            self.nn_fit = None

        n_samples = self.X_.shape[0]

        affinity_matrix = self._get_kernel(self.X_)

        # compute laplacian matrix

        laplacian = csgraph.laplacian(affinity_matrix, normed=True)

        laplacian_matrix = -laplacian

        if sparse.isspmatrix(laplacian_matrix):
            diag_mask = (laplacian_matrix.row == laplacian_matrix.col)
            laplacian_matrix.data[diag_mask] = 0.0

        else:
            laplacian_matrix.flat[::n_samples + 1] = 0.0  # set diagonal to 0.0

        return laplacian_matrix


if __name__ == "__main__":

    ''' Prepare data examples '''
    examples_num = 200
    X, y = make_circles(n_samples=examples_num, shuffle=False)
    labels = np.full(examples_num, -1)

    ''' Set the example of label 0 and 1 '''
    labels[0] = 0
    labels[-1] = 1

    ''' Learn the data structure with label spreading algorithm '''
    lsa = LabelSpreadingAlgorithm(alpha=0.8, max_iter=500)
    lsa.fit(X, labels)

    spread_labels = lsa.transduction_

    ''' Plot output labels '''

    plt.figure(figsize=(10.8, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X[labels == 0, 0], X[labels == 0, 1], color='r',
                marker='*', lw=0, label="label 0", s=40)
    plt.scatter(X[labels == 1, 0], X[labels == 1, 1], color='g',
                marker='*', lw=0, label='label 1', s=40)
    plt.scatter(X[labels == -1, 0], X[labels == -1, 1], color='darkgrey',
                marker='.', label='unlabeled')
    plt.legend(scatterpoints=1, shadow=False, loc='upper right')
    plt.title("Raw data examples")

    plt.subplot(1, 2, 2)
    spread_label_array = np.asarray(spread_labels)
    n_label_0 = np.where(spread_label_array == 0)[0]
    n_label_1= np.where(spread_label_array == 1)[0]
    plt.scatter(X[n_label_0, 0], X[n_label_0, 1], color='r',
                marker='*', lw=0, s=40, label="label 0 learned")
    plt.scatter(X[n_label_1, 0], X[n_label_1, 1], color='g',
                marker='*', lw=0, s=40, label="label 1 learned")
    plt.legend(scatterpoints=1, shadow=False, loc='upper right')
    plt.title("Labels learned with Label Spreading")

    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.9, top=0.9)
    plt.show()
