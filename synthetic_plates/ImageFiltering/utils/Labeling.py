import math

import numpy as np
import pandas as pd
from synthetic_plates.ImageFiltering.utils.NEATER import NEATER
from imblearn.over_sampling import RandomOverSampler


class Labeling:
    def __init__(self, train_data_file, no_label_data_file):
        train_data = pd.read_csv(train_data_file)
        features = train_data.columns.drop(['instance_name', 'label'])
        self.X = train_data[features].to_numpy()
        self.y = train_data['label'].replace(['Correct', 'Corrupt'], [1, 0]).to_numpy()
        no_label_data = pd.read_csv(no_label_data_file)
        self.X_no_label = no_label_data[features].to_numpy()
        self.y_no_label = no_label_data['label'].to_numpy()

    def neater(self):
        print('0 class in train oversampled: %d' % np.sum(self.y == 0))
        print('1 class in train oversampled: %d' % np.sum(self.y == 1))

        # balancing
        oversampling = RandomOverSampler(sampling_strategy=1.0)
        # fit and apply the transform
        x, y = oversampling.fit_resample(self.X, self.y)

        print('0 class in train oversampled: %d' % np.sum(y == 0))
        print('1 class in train oversampled: %d' % np.sum(y == 1))

        y_prediction = []
        train_len = len(self.X)
        itr = math.ceil(len(self.X_no_label) / train_len)
        for i in range(itr):
            print(i+1, "/", itr)
            x_n = self.X_no_label[i * train_len:i * train_len + train_len]
            y_n = self.y_no_label[i * train_len:i * train_len + train_len]

            neater = NEATER(alpha=0.999999, h=20)
            x_prediction, prediction = neater.sample(x, y, x_n, y_n, data_produced=0)
            y_prediction = np.append(np.array(y_prediction), prediction)

        return np.array(y_prediction)

    def run(self, method):
        return self.X_no_label, getattr(self, method)()
