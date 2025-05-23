import numpy as np

from synthetic_plates.ImageFiltering.utils._metric_tensor import (NearestNeighborsWithMetricTensor,
                                                                  pairwise_distances_mahalanobis)
from synthetic_plates.ImageFiltering.utils._OverSampling import OverSampling
from synthetic_plates.ImageFiltering.utils._SMOTE import SMOTE
from synthetic_plates.ImageFiltering.utils._ADASYN import ADASYN

from synthetic_plates.ImageFiltering.utils._logger import logger

from sklearn.neighbors import DistanceMetric

_logger = logger

__all__ = ['NEATER']


class GERISPRIME(OverSampling):
    """
    References:
        * BibTex::

            @INPROCEEDINGS{neater,
                            author={Almogahed, B. A. and Kakadiaris, I. A.},
                            booktitle={2014 22nd International Conference on
                                         Pattern Recognition},
                            title={NEATER: Filtering of Over-sampled Data
                                    Using Non-cooperative Game Theory},
                            year={2014},
                            volume={},
                            number={},
                            pages={1371-1376},
                            keywords={no_labels handling;game theory;information
                                        filtering;NEATER;imbalanced no_labels
                                        problem;synthetic no_labels;filtering of
                                        over-sampled no_labels using non-cooperative
                                        game theory;Games;Game theory;Vectors;
                                        Sociology;Statistics;Silicon;
                                        Mathematical model},
                            doi={10.1109/ICPR.2014.245},
                            ISSN={1051-4651},
                            month={Aug}}

    Notes:
        * Evolving both majority and minority probabilities as nothing ensures
            that the probabilities remain in the range [0,1], and they need to
            be normalized.
        * The inversely weighted function needs to be cut at some value (like
            the alpha level), otherwise it will overemphasize the utility of
            having differing neighbors next to each other.
    """

    categories = [OverSampling.cat_extensive,
                  OverSampling.cat_borderline,
                  OverSampling.cat_changes_majority,
                  OverSampling.cat_metric_learning]

    def __init__(self,
                 proportion=1.0,
                 smote_n_neighbors=20,
                 *,
                 nn_params={},
                 b=40,
                 alpha=0.999999,
                 h=20,
                 n_jobs=1,
                 random_state=None):
        """
        Constructor of the sampling object

        Args:
            proportion (float): proportion of the difference of n_maj and n_min
                                to sample e.g. 1.0 means that after sampling
                                the number of minority samples will be equal to
                                the number of majority samples
            smote_n_neighbors (int): number of neighbors in SMOTE sampling
            nn_params (dict): additional parameters for nearest neighbor calculations, any
                                parameter NearestNeighbors accepts, and additionally use
                                {'metric': 'precomputed', 'metric_learning': '<method>', ...}
                                with <method> in 'ITML', 'LSML' to enable the learning of
                                the metric to be used for neighborhood calculations
            b (int): number of neighbors
            alpha (float): smoothing term
            h (int): number of iterations in evolution
            n_jobs (int): number of parallel jobs
            random_state (int/RandomState/None): initializer of random_state,
                                                    like in sklearn
        """
        super().__init__()
        self.check_greater_or_equal(proportion, "proportion", 0)
        self.check_greater_or_equal(smote_n_neighbors, "smote_n_neighbors", 1)
        self.check_greater_or_equal(b, "b", 1)
        self.check_greater_or_equal(alpha, "alpha", 0)
        self.check_greater_or_equal(h, "h", 0)
        self.check_n_jobs(n_jobs, 'n_jobs')

        self.proportion = proportion
        self.smote_n_neighbors = smote_n_neighbors
        self.nn_params = nn_params
        self.b = b
        self.alpha = alpha
        self.h = h
        self.n_jobs = n_jobs

        self.set_random_state(random_state)

    # @ classmethod
    # def parameter_combinations(cls, raw=False):
    #     """
    #     Generates reasonable parameter combinations.
    #
    #     Returns:
    #         list(dict): a list of meaningful parameter combinations
    #     """
    #     parameter_combinations = {'proportion': [0.1, 0.25, 0.5, 0.75,
    #                                              1.0, 1.5, 2.0],
    #                               'smote_n_neighbors': [3, 5, 7],
    #                               'b': [3, 5, 7],
    #                               'alpha': [0.9999],
    #                               'h': [20]}
    #     return cls.generate_parameter_combinations(parameter_combinations, raw)

    def sample(self, X, y, X_test, y_test, data_produced, metric='minkowski'):
        """
        Does the sample generation according to the class parameters.

        Args:
            X (np.ndarray): training set
            y (np.array): target labels

        Returns:
            (np.ndarray, np.array): the extended training set and target labels
        """
        self.data_producd = data_produced
        self.X_test = X_test
        self.y_test = y_test

        _logger.info(self.__class__.__name__ + ": " +
                     "Running sampling via %s" % self.descriptor())

        self.class_label_statistics(X, y)

        if not self.check_enough_min_samples_for_sampling():
            return X.copy(), y.copy()

        nn_params = {**self.nn_params}
        nn_params['metric_tensor'] = self.metric_tensor_from_nn_params(nn_params, X, y)

        if data_produced == 1:

            # Applying SMOTE and ADASYN
            X_0, y_0 = SMOTE(proportion=self.proportion,
                             n_neighbors=self.smote_n_neighbors,
                             nn_params=nn_params,
                             n_jobs=self.n_jobs,
                             random_state=self._random_state_init).sample(X, y)

            X_1, y_1 = ADASYN(n_neighbors=self.b,
                              nn_params=nn_params,
                              n_jobs=self.n_jobs,
                              random_state=self._random_state_init).sample(X, y)

            X_new = np.vstack([X_0, X_1[len(X):]])
            y_new = np.hstack([y_0, y_1[len(y):]])

            X_syn = X_new[len(X):]

            if len(X_syn) == 0:
                return X.copy(), y.copy()

        if data_produced == 0:
            X_new = np.concatenate((X, X_test), axis=0)
            y_test = np.zeros(y_test.shape)
            y_new = np.concatenate((y, y_test), axis=0)
            X_syn = X_test

        X_all = X_new
        y_all = y_new

        cumulative_corrupt = np.zeros(len(X_all))
        cumulative_correct = np.zeros(len(X_all))

        # binary indicator indicating synthetic instances
        synthetic = np.hstack(
            [np.array([False] * len(X)), np.array([True] * len(X_syn))])

        # initializing strategy probabilities
        prob = np.zeros(shape=(len(X_all), 2))
        prob.fill(0.5)
        for i in range(len(X)):
            if y[i] == self.min_label:
                prob[i, 0], prob[i, 1] = 0.0, 1.0
            else:
                prob[i, 0], prob[i, 1] = 1.0, 0.0

        # Finding nearest neighbors, +1 as X_syn is part of X_all and nearest
        # neighbors will be themselves
        if metric == 'mahalanobis':
            nn = NearestNeighborsWithMetricTensor(n_neighbors=self.b + 1,
                                                  n_jobs=self.n_jobs,
                                                  **nn_params)
        else:
            nn = NearestNeighborsWithMetricTensor(n_neighbors=self.b + 1,
                                                  n_jobs=self.n_jobs,
                                                  metric=metric,
                                                  **nn_params)
        nn.fit(X_all)
        indices = nn.kneighbors(X_syn, return_distance=False)

        # computing distances
        if metric == 'mahalanobis':
            dm = pairwise_distances_mahalanobis(X_all,
                                                X_syn,
                                                nn_params.get('metric_tensor', None))
        else:
            dist = DistanceMetric.get_metric(metric)
            dm = dist.pairwise(X_syn, X_all)
        dm[dm == 0] = 1e-8
        dm = 1 / dm

        dm[dm > 0.01] = 0.01

        def wprob_mixed(prob, i):
            ind = indices[i][1:]
            a_11 = dm[i, ind]
            # term_0 = np.multiply(a_11 * prob[i][0] * prob[ind, 0],  np.maximum(1-prob[ind , 0],0.1))
            term_0 = a_11 * prob[i][0] * prob[ind, 0]
            term_1 = 0 * (prob[i][1] * prob[ind, 0] +
                          prob[i][0] * prob[ind, 1])
            term_2 = a_11 * prob[i][1] * prob[ind, 1]
            return np.sum(term_0 + term_1 + term_2)
        def wprob_min(prob, i):

            term_0 = 0 * prob[indices[i][1:], 0]
            term_1 = 0 * (1 * prob[indices[i][1:], 0] +
                          0 * prob[indices[i][1:], 1])
            term_2 = dm[i, indices[i][1:]] * prob[indices[i][1:], 1]
            return np.sum(term_0 + term_1 + term_2)

        def wprob_maj(prob, i):
            term_0 = dm[i, indices[i][1:]] * prob[indices[i][1:], 0]
            term_1 = 0 * (0 * prob[indices[i][1:], 0] +
                          1 * prob[indices[i][1:], 1])
            term_2 = 0 * prob[indices[i][1:], 1]
            return np.sum(term_0 + term_1 + term_2)

        def utilities(prob):
            """
            Computes the utilit function

            Args:
                prob (np.matrix): strategy probabilities

            Returns:
                np.array, np.array, np.array: utility values, minority
                                                utilities, majority
                                                utilities
            """

            domain = range(len(X_syn))
            util_mixed = np.array([wprob_mixed(prob, i) for i in domain])
            util_mixed = np.hstack([np.array([0] * len(X)), util_mixed])

            util_min = np.array([wprob_min(prob, i) for i in domain])
            util_min = np.hstack([np.array([0] * len(X)), util_min])

            util_maj = np.array([wprob_maj(prob, i) for i in domain])
            util_maj = np.hstack([np.array([0] * len(X)), util_maj])

            return util_min, util_maj , util_mixed



        def evolution( prob, synthetic):
            """
            Executing one step of the probabilistic evolution

            Args:
                prob (np.matrix): strategy probabilities
                synthetic (np.array): flags of synthetic examples
                alpha (float): smoothing function

            Returns:
                np.matrix: updated probabilities
            """
            util_min, util_maj , util_mixed = utilities(prob)
            r_corrupt = util_maj - util_mixed
            r_correct = util_min - util_mixed
            prob_new = prob.copy()

            r_corrupt[r_corrupt < 0] = 0
            r_correct[r_correct < 0] = 0
            R1 = r_correct
            R2 = r_corrupt

            synthetic_values = R1/(R1+R2)
            prob_new[:, 1] = np.where(synthetic, synthetic_values, prob[:, 1])

            synthetic_values = R2/(R1+R2)
            prob_new[:, 0] = np.where(synthetic, synthetic_values, prob[:, 0])

            norm_factor = np.sum(prob_new, axis=1)

            prob_new[:, 0] = prob_new[:, 0] / norm_factor
            prob_new[:, 1] = prob_new[:, 1] / norm_factor

            return prob_new , r_correct , r_corrupt

        # executing the evolution


        prob , _ , _ = evolution( prob, synthetic)

        # determining final labels
        y_all[len(X):] = np.argmax(prob[len(X):], axis=1)

        return X_all[len(X):], y_all[len(X):]

    def get_params(self, deep=False):
        """
        Returns:
            dict: the parameters of the current sampling object
        """
        return {'proportion': self.proportion,
                'smote_n_neighbors': self.smote_n_neighbors,
                'b': self.b,
                'alpha': self.alpha,
                'h': self.h,
                'nn_params': self.nn_params,
                'n_jobs': self.n_jobs,
                'random_state': self._random_state_init}
