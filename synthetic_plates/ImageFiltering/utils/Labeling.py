import math

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from synthetic_plates.ImageFiltering.utils.NEATER import NEATER
from imblearn.over_sampling import RandomOverSampler


class Labeling:
    def __init__(self, train_data_file, no_label_data_file):
        train_data = pd.read_csv(train_data_file)

        features = train_data.columns.drop(['instance_name', 'label', 'label_noise', 'label_rotation'])
        rotation_features = ['yaw', 'pitch', 'roll']
        noise_features = features.drop(rotation_features)

        self.X = train_data[features].to_numpy()
        self.X_noise = train_data[noise_features].to_numpy()
        self.X_rotation = train_data[rotation_features].to_numpy()

        self.y = train_data['label'].replace(['Correct', 'Corrupt'], [1, 0]).to_numpy()
        self.y_noise = train_data['label_noise'].replace(['Correct', 'Corrupt'], [1, 0]).to_numpy()
        self.y_rotation = train_data['label_rotation'].replace(['Correct', 'Corrupt'], [1, 0]).to_numpy()

        no_label_data = pd.read_csv(no_label_data_file)

        self.X_no_label = no_label_data[features].to_numpy()
        self.X_no_label_noise = no_label_data[noise_features].to_numpy()
        self.X_no_label_rotation = no_label_data[rotation_features].to_numpy()

        self.y_no_label = no_label_data['label'].replace(['Correct', 'Corrupt'], [1, 0]).to_numpy()
        self.y_no_label_noise = no_label_data['label_noise'].replace(['Correct', 'Corrupt'], [1, 0]).to_numpy()
        self.y_no_label_rotation = no_label_data['label_rotation'].replace(['Correct', 'Corrupt'], [1, 0]).to_numpy()

    def neater(self):
        # balancing
        oversampling = RandomOverSampler(sampling_strategy=1.0)
        # fit and apply the transform
        x_noise, y_noise = oversampling.fit_resample(self.X_noise, self.y_noise)
        x_rotation, y_rotation = oversampling.fit_resample(self.X_rotation, self.y_rotation)

        y_prediction_rotation = []
        y_prediction_noise = []
        train_len = len(self.X)
        itr = math.ceil(len(self.X_no_label) / train_len)
        for i in range(itr):
            print(i + 1, "/", itr)

            x_n = self.X_no_label_rotation[i * train_len:i * train_len + train_len]
            y_n = self.y_no_label_rotation[i * train_len:i * train_len + train_len]

            neater = NEATER(alpha=0.999999, h=20)
            x_prediction, prediction = neater.sample(x_rotation, y_rotation, x_n, y_n, data_produced=0, metric='mahalanobis')
            y_prediction_rotation = np.append(np.array(y_prediction_rotation), prediction)

            x_n = self.X_no_label_noise[i * train_len:i * train_len + train_len]
            y_n = self.y_no_label_noise[i * train_len:i * train_len + train_len]

            neater = NEATER(alpha=0.999999, h=20)
            x_prediction, prediction = neater.sample(x_noise, y_noise, x_n, y_n, data_produced=0, metric='mahalanobis')
            y_prediction_noise = np.append(np.array(y_prediction_noise), prediction)

        return np.array(y_prediction_rotation) * np.array(y_prediction_noise)

    def nb(self):
        clf = GaussianNB(var_smoothing=0.01519911082952933).fit(self.X_noise, self.y_noise)
        pred_noise = clf.predict(self.X_no_label_noise)
        clf = GaussianNB(var_smoothing=0.01519911082952933).fit(self.X_rotation, self.y_rotation)
        pred_rotation = clf.predict(self.X_no_label_rotation)
        return pred_noise * pred_rotation

    def run(self, method):
        return self.X_no_label, getattr(self, method)()
