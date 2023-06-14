import numpy as np

from synthetic_plates.ImageFiltering.utils._metric_tensor import (NearestNeighborsWithMetricTensor,
                                                                  pairwise_distances_mahalanobis)
from synthetic_plates.ImageFiltering.utils._OverSampling import OverSampling
from synthetic_plates.ImageFiltering.utils._SMOTE import SMOTE
from synthetic_plates.ImageFiltering.utils._ADASYN import ADASYN

from synthetic_plates.ImageFiltering.utils._logger import logger

from sklearn.neighbors import DistanceMetric

class NEATERJUNIOR(OverSampling):
    def __init__(self, synthetic, indices , dm , labels , alpha ):
        super().__init__()
        self.synthetic = synthetic
        self.indices = indices
        self.dm = dm
        self.ndm = np.zeros((len(dm)  , len(indices[0]) - 1))
        self.labels = labels
        self.alpha = alpha
        for i in range(0 , len(dm)):
            ind = indices[i][1:]
            self.ndm[i] = self.dm[i][ind]

    def probabilityOfCorrect(self):
        for player in range(0 , len(self.synthetic)):
            u1 = 0
            u2 = 0
            neighbours = self.indices[player]
            for i in range(1 , len(neighbours)):
                u1 = u1 + self.labels[self.indices[player][i]] * self.ndm[player][i-1] * self.labels[self.indices[player][0]]+ \
                     ( 1 - self.labels[self.indices[player][i]]) * self.ndm[player][i-1] * (1 -self.labels[self.indices[player][0]])
                u2 = u2 + self.labels[self.indices[player][i]]* self.ndm[player][i-1] * 1
            probNew = ((self.alpha + u2)/(self.alpha + u1))*self.labels[self.indices[player][0]]
            self.labels[self.indices[player][0]] =probNew
        return self.labels