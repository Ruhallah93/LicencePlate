import numpy as np
class NoRegretJunior():
    def __init__(self, synthetic, indices , dm , labels, cumulativeRegretCorrect ,cumulativeRegretCorrupt ):
        super().__init__()
        self.synthetic = synthetic
        self.indices = indices
        self.dm = dm
        self.ndm = np.zeros((len(dm)  , len(indices[0]) - 1))
        self.labels = labels
        self.cumulativeRegretCorrect = cumulativeRegretCorrect
        self.cumulativeRegretCorrupt = cumulativeRegretCorrupt
        for i in range(0 , len(dm)):
            ind = indices[i][1:]
            self.ndm[i] = self.dm[i][ind]

    def probabilityOfCorrect(self):
        for player in range(0 , len(self.synthetic)):
            pure_correct = 0
            pure_corrupt = 0
            mixed = 0
            neighbours = self.indices[player]
            R1 = 0
            R2 = 0
            for i in range(1 , len(neighbours)):
                mixed = mixed + self.labels[self.indices[player][i]] * self.ndm[player][i-1] * self.labels[self.indices[player][0]]+ \
                     ( 1 - self.labels[self.indices[player][i]]) * self.ndm[player][i-1] * (1 -self.labels[self.indices[player][0]])

                pure_correct = pure_correct + self.labels[self.indices[player][i]]* self.ndm[player][i-1] * 1
                pure_corrupt = pure_corrupt + (1 - self.labels[self.indices[player][i]]) * self.ndm[player][i - 1] * 1

            regret_correct = pure_correct - mixed
            regret_corrupt = pure_corrupt - mixed
            self.cumulativeRegretCorrect[player] = self.cumulativeRegretCorrect[player] + regret_correct
            self.cumulativeRegretCorrupt[player] = self.cumulativeRegretCorrupt[player] + regret_corrupt
            if self.cumulativeRegretCorrect[player] < 0:
                R1 = 0
                R2 = self.cumulativeRegretCorrupt[player]
            if self.cumulativeRegretCorrupt[player] < 0:
                R1 = self.cumulativeRegretCorrect[player]
                R2 = 0
            probNew = R1 / (R1 + R2)
            self.labels[self.indices[player][0]] = probNew
        return self.labels , self.cumulativeRegretCorrect , self.cumulativeRegretCorrupt




