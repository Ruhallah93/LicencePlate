import random

from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

from synthetic_plates.ImageFiltering.utils.NEATER import NEATER
from synthetic_plates.ImageFiltering.utils.GERIS import GERIS
from synthetic_plates.ImageFiltering.utils.GERISPRIME import GERISPRIME
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from scipy import stats as st
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


class Evaluation:
    def __init__(self, test_size, test_file):
        self.test_size = test_size
        # reader = Reader(test_file)
        # self.X, self.y, self.header = reader.construct()

        train_data = pd.read_csv(test_file)
        features = train_data.columns.drop(['instance_name', 'label', 'label_noise', 'label_rotation'])
        rotation_features = ['yaw', 'pitch', 'roll']
        noise_features = features.drop(rotation_features)

        self.X = train_data[features].to_numpy()
        self.X_noise = train_data[noise_features].to_numpy()
        self.X_rotation = train_data[rotation_features].to_numpy()

        self.y = train_data['label'].replace(['Correct', 'Corrupt'], [1, 0]).to_numpy()
        self.y_noise = train_data['label_noise'].replace(['Correct', 'Corrupt'], [1, 0]).to_numpy()
        self.y_rotation = train_data['label_rotation'].replace(['Correct', 'Corrupt'], [1, 0]).to_numpy()

        print('0 class in test: %d' % np.sum(self.y == 0))
        print('1 class in test: %d' % np.sum(self.y == 1))
        self.X_train_noise, self.X_test_noise, \
        self.X_train_rotation, self.X_test_rotation, \
        self.X_train, self.X_test, \
        self.y_train_noise, self.y_test_noise, \
        self.y_train_rotation, self.y_test_rotation, \
        self.y_train, self.y_test = train_test_split(self.X_noise, self.X_rotation, self.X,
                                                     self.y_noise, self.y_rotation, self.y,
                                                     test_size=test_size, shuffle=True)

    def mlp(self, X_train, X_test, y_train, y_test):
        # grid = {
        #     'hidden_layer_sizes': [(100,), (200,), (300,), (400,), (500,)],
        #     'activation': ['identity', 'logistic', 'relu'],
        #     'solver': ['sgd', 'adam'],
        #     'alpha': [0.1, 0.01, 0.001, 0.0001, 0.05],
        #     'learning_rate': ['constant', 'adaptive'],
        #     'max_iter': [500, 1000, 2000]
        # }
        # classifier = MLPClassifier()
        # rf_cv = GridSearchCV(estimator=classifier, param_grid=grid, cv=5)
        # rf_cv.fit(self.X_train, self.y_train)
        # print(rf_cv.best_params_)

        # alpha = 0.0001
        clf = MLPClassifier(activation='relu', alpha=0.1, hidden_layer_sizes=(300,), learning_rate='adaptive',
                            max_iter=2000, solver='sgd').fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    def svm(self, X_train, X_test, y_train, y_test):
        # grid = {'C': [0.1, 1, 10, 100, 1000],
        #         'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        #         'kernel': ['rbf', 'poly', 'sigmoid']}
        # classifier = SVC()
        # rf_cv = GridSearchCV(estimator=classifier, param_grid=grid, cv=5)
        # rf_cv.fit(self.X_train, self.y_train)
        # print(rf_cv.best_params_)

        clf = SVC(gamma=0.001, C=100, kernel='rbf').fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    def nb(self, X_train, X_test, y_train, y_test):
        # grid = {'var_smoothing': np.logspace(0, -9, num=100)}
        # classifier = GaussianNB()
        # rf_cv = GridSearchCV(estimator=classifier, param_grid=grid, cv=10)
        # rf_cv.fit(self.X_train, self.y_train)
        # print(rf_cv.best_params_)

        clf = GaussianNB(var_smoothing=0.01519911082952933).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    def rf(self, X_train, X_test, y_train, y_test):
        # grid = {
        #     'n_estimators': [200, 300, 400, 500],
        #     'max_features': ['sqrt', 'log2'],
        #     'max_depth': [4, 5, 6, 7, 8],
        #     'criterion': ['gini', 'entropy'],
        #     'random_state': [18, 42]
        # }
        # classifier = RandomForestClassifier()
        # rf_cv = GridSearchCV(estimator=classifier, param_grid=grid, cv=10)
        # rf_cv.fit(self.X_train, self.y_train)
        # print(rf_cv.best_params_)

        # clf = RandomForestClassifier(criterion='entropy', max_depth=7, random_state=18,
        #                              max_features='log2', n_estimators=200).fit(self.X_train, self.y_train)
        clf = RandomForestClassifier(criterion='gini', max_depth=4, random_state=42,
                                     max_features='log2', n_estimators=500).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    def knn(self, X_train, X_test, y_train, y_test):
        # grid = {
        #     'n_neighbors': list(range(1, 31)),
        # }
        # classifier = KNeighborsClassifier()
        # rf_cv = GridSearchCV(estimator=classifier, param_grid=grid, cv=10)
        # rf_cv.fit(self.X_train, self.y_train)
        # print(rf_cv.best_params_)

        clf = KNeighborsClassifier(n_neighbors=14).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_pred

    def neater(self, X_train, X_test, y_train, y_test, h, b, alpha, sampling_strategy, metric):
        # Model Selection
        # best_score = 0
        # best_param = []
        # for ss in range(5, 11, 1):
        #     print("Balancing...")
        #     oversample = RandomOverSampler(sampling_strategy=ss / 10)
        #     # fit and apply the transform
        #     X_train_tmp, y_train_tmp = oversample.fit_resample(X_train, y_train)
        #
        #     for b in range(2, 40):
        #         for h in range(2, 40):
        #             for alpha in [1, 0.999999, 0.9, 0.8, 0.5]:
        #                 neater = NEATER(alpha=alpha, h=h, b=b)
        #                 X_pred, y_pred = neater.sample(X_train_tmp, y_train_tmp, X_test, y_test, data_produced=0)
        #                 r = recall_score(y_true=y_test, y_pred=y_pred, average=None)[0]
        #                 print("current score:", r, "param:", [h, b, alpha, ss / 10])
        #                 if best_score < r:
        #                     best_score = r
        #                     best_param = [h, b, alpha, ss / 10]
        # print("best score:", best_score)
        # print("best param:", best_param)

        # print('0 class in train oversampled: %d' % np.sum(y_train == 0))
        # print('1 class in train oversampled: %d' % np.sum(y_train == 1))

        if sampling_strategy > 0:
            print("Balancing...")
            oversample = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_train, y_train = oversample.fit_resample(X_train, y_train)

        # print('0 class in train oversampled: %d' % np.sum(y_train == 0))
        # print('1 class in train oversampled: %d' % np.sum(y_train == 1))

        neater = NEATER(alpha=alpha, h=h, b=b)
        X_pred, y_pred = neater.sample(X_train, y_train, X_test, y_test, data_produced=0, metric=metric)

        return y_pred

    def geris(self, X_train, X_test, y_train, y_test, h, b, alpha, sampling_strategy, metric):

        if sampling_strategy > 0:
            print("Balancing...")
            oversample = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_train, y_train = oversample.fit_resample(X_train, y_train)

        # print('0 class in train oversampled: %d' % np.sum(y_train == 0))
        # print('1 class in train oversampled: %d' % np.sum(y_train == 1))

        geris = GERIS(alpha=alpha, h=h, b=b)
        X_pred, y_pred = geris.sample(X_train, y_train, X_test, y_test, data_produced=0, metric=metric)

        return y_pred
    def geris_regret(self, X_train, X_test, y_train, y_test, h, b, alpha, sampling_strategy, metric):

        if sampling_strategy > 0:
            print("Balancing...")
            oversample = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_train, y_train = oversample.fit_resample(X_train, y_train)

        # print('0 class in train oversampled: %d' % np.sum(y_train == 0))
        # print('1 class in train oversampled: %d' % np.sum(y_train == 1))

        geris_regret = GERISPRIME(alpha=alpha, h=h, b=b)
        X_pred, y_pred = geris_regret.sample(X_train, y_train, X_test, y_test, data_produced=0, metric=metric)

        return y_pred
    def voting(self, X_train, X_test, y_train, y_test):
        _, _, _, y_test, y_pred1 = self.run('mlp')
        _, _, _, y_test, y_pred2 = self.run('nb')
        _, _, _, y_test, y_pred3 = self.run('neater')
        _, _, _, y_test, y_pred4 = self.run('rf')
        _, _, _, y_test, y_pred5 = self.run('svm')
        _, _, _, y_test, y_pred6 = self.run('knn')

        # k = random.randint(3, 6)
        # y_rand = random.choices([y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6], k=k)

        mode, count = st.mode([y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6])
        y_pred = mode[0]

        return y_pred

    def run(self, method):
        if method == 'neater':
            # for i in range(4, 11):
            #     for j in range(4, 11):
            #         y_pred_noises = self.neater(self.X_train_noise, self.X_test_noise,
            #                                     self.y_train_noise, self.y_test_noise,
            #                                     h=6, alpha=1, sampling_strategy=i / 10)
            #         y_pred_rotations = self.neater(self.X_train_rotation, self.X_test_rotation,
            #                                        self.y_train_rotation, self.y_test_rotation,
            #                                        h=4, alpha=1, sampling_strategy=j / 10)
            #         y_pred = y_pred_noises * y_pred_rotations
            #         print(i, j, recall_score(y_true=self.y_test, y_pred=y_pred, average=None)[0])

            y_pred_noises = self.neater(self.X_train_noise, self.X_test_noise,
                                        self.y_train_noise, self.y_test_noise,
                                        h=10, b=40, alpha=1, sampling_strategy=1,
                                        metric='euclidean')
            y_pred_rotations = self.neater(self.X_train_rotation, self.X_test_rotation,
                                           self.y_train_rotation, self.y_test_rotation,
                                           h=10, b=40, alpha=1, sampling_strategy=0.6,
                                           metric='euclidean')
        elif method == 'geris':
            # for i in range(4, 11):
            #     for j in range(4, 11):
            #         y_pred_noises = self.neater(self.X_train_noise, self.X_test_noise,
            #                                     self.y_train_noise, self.y_test_noise,
            #                                     h=6, alpha=1, sampling_strategy=i / 10)
            #         y_pred_rotations = self.neater(self.X_train_rotation, self.X_test_rotation,
            #                                        self.y_train_rotation, self.y_test_rotation,
            #                                        h=4, alpha=1, sampling_strategy=j / 10)
            #         y_pred = y_pred_noises * y_pred_rotations
            #         print(i, j, recall_score(y_true=self.y_test, y_pred=y_pred, average=None)[0])

            y_pred_noises = self.geris(self.X_train_noise, self.X_test_noise,
                                        self.y_train_noise, self.y_test_noise,
                                        h=4, b=40, alpha=0.9, sampling_strategy=1,
                                        metric='euclidean')
            y_pred_rotations = self.geris(self.X_train_rotation, self.X_test_rotation,
                                           self.y_train_rotation, self.y_test_rotation,
                                           h=4, b=40, alpha=0.9, sampling_strategy=0.6,
                                           metric='euclidean')
        else:
            y_pred_noises = getattr(self, method)(self.X_train_noise, self.X_test_noise,
                                                  self.y_train_noise, self.y_test_noise)
            y_pred_rotations = getattr(self, method)(self.X_train_rotation, self.X_test_rotation,
                                                     self.y_train_rotation, self.y_test_rotation)
        return self.X_train, self.y_train, self.X_test, self.y_test, self.y_test_noise, self.y_test_rotation, y_pred_noises, y_pred_rotations
