import math

import numpy as np
class CumulativePayoffJunior():
    def __init__(self, synthetic, indices , dm , labels, alpha , cumulativePayoffCorrect ,cumulativePayoffCorrupt ):
        super().__init__()
        self.synthetic = synthetic
        self.indices = indices
        self.dm = dm
        self.ndm = np.zeros((len(dm)  , len(indices[0]) - 1))
        self.labels = labels
        self.alpha = alpha
        self.cumulativePayoffCorrect = cumulativePayoffCorrect
        self.cumulativePayoffCorrupt = cumulativePayoffCorrupt
        for i in range(0 , len(dm)):
            ind = indices[i][1:]
            self.ndm[i] = self.dm[i][ind]

    def probabilityOfCorrect(self):
        for player in range(0 , len(self.synthetic)):
            pure_correct = 0
            pure_corrupt = 0
            neighbours = self.indices[player]
            R1 = 0
            R2 = 0
            for i in range(1 , len(neighbours)):
                pure_correct = pure_correct + self.labels[self.indices[player][i]]* self.ndm[player][i-1] * 1
                pure_corrupt = pure_corrupt + (1 - self.labels[self.indices[player][i]]) * self.ndm[player][i - 1] * 1
            self.cumulativePayoffCorrect[player] = self.cumulativePayoffCorrect[player] + pure_correct
            self.cumulativePayoffCorrupt[player] = self.cumulativePayoffCorrupt[player] + pure_corrupt
            R1 = math.pow((1 + self.alpha) , self.cumulativePayoffCorrect[player])
            R2 = math.pow((1 + self.alpha) , self.cumulativePayoffCorrupt[player])
            probNew = R1 / (R1 + R2)
            self.labels[self.indices[player][0]] = probNew
        return self.labels , self.cumulativePayoffCorrect , self.cumulativePayoffCorrupt




