from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np

from ImageFiltering.utils.NEATER import NEATER
from ImageFiltering.utils.Reader import Reader
from ImageFiltering.utils.Writer import Writer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from scipy import stats as st
from sklearn.model_selection import GridSearchCV


class Test:
    def __init__(self, test_size, test_file):
        self.test_size = test_size
        self.test_file = test_file

        reader = Reader(test_file)
        self.X, self.y, self.header = reader.construct()
        print('0 class in test: %d' % np.sum(self.y == 0))
        print('1 class in test: %d' % np.sum(self.y == 1))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size
                                                                                , shuffle=True
                                                                                , random_state=42)

    def mlp(self):
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

        clf = MLPClassifier(activation='identity', alpha=0.05, hidden_layer_sizes=(500,), learning_rate='adaptive',
                            max_iter=1000, solver='adam').fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        print("with MLP:")
        print(confusion_matrix(y_true=self.y_test, y_pred=y_pred))
        print(classification_report(self.y_test, y_pred))
        return y_pred

    def svm(self):
        # grid = {'C': [0.1, 1, 10, 100, 1000],
        #         'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        #         'kernel': ['rbf', 'poly', 'sigmoid']}
        # classifier = SVC()
        # rf_cv = GridSearchCV(estimator=classifier, param_grid=grid, cv=5)
        # rf_cv.fit(self.X_train, self.y_train)
        # print(rf_cv.best_params_)

        clf = SVC(gamma=0.0001, C=0.1, kernel='sigmoid').fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        print("with SVC:")
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(y_true=self.y_test, y_pred=y_pred))
        return y_pred

    def nb(self):
        # grid = {'var_smoothing': np.logspace(0, -9, num=100)}
        # classifier = GaussianNB()
        # rf_cv = GridSearchCV(estimator=classifier, param_grid=grid, cv=10)
        # rf_cv.fit(self.X_train, self.y_train)
        # print(rf_cv.best_params_)

        gnb = GaussianNB(var_smoothing=0.01).fit(self.X_train, self.y_train)
        y_pred = gnb.predict(self.X_test)
        print("with NB:")
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(y_true=self.y_test, y_pred=y_pred))
        return y_pred

    def rf(self):
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
        clf = RandomForestClassifier(criterion='gini', max_depth=5, random_state=42,
                                     max_features='sqrt', n_estimators=400).fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        print("with RF:")
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(y_true=self.y_test, y_pred=y_pred))
        return y_pred

    def knn(self):
        # grid = {
        #     'n_neighbors': list(range(1, 31)),
        # }
        # classifier = KNeighborsClassifier()
        # rf_cv = GridSearchCV(estimator=classifier, param_grid=grid, cv=10)
        # rf_cv.fit(self.X_train, self.y_train)
        # print(rf_cv.best_params_)

        clf = KNeighborsClassifier(n_neighbors=6).fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        print("with KNN:")
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(y_true=self.y_test, y_pred=y_pred))
        return y_pred

    def neater(self):
        print('0 class in train oversampled: %d' % np.sum(self.y_train == 0))
        print('1 class in train oversampled: %d' % np.sum(self.y_train == 1))

        print("Balancing...")
        oversample = RandomOverSampler(sampling_strategy=1)
        # fit and apply the transform
        X_train, y_train = oversample.fit_resample(self.X_train, self.y_train)

        print('0 class in train oversampled: %d' % np.sum(y_train == 0))
        print('1 class in train oversampled: %d' % np.sum(y_train == 1))

        n = NEATER()
        X_pred, y_pred = n.sample(X_train, y_train, self.X_test, self.y_test, data_produced=0)

        print("with NEATER:")
        print(confusion_matrix(y_true=self.y_test, y_pred=y_pred))
        print(classification_report(y_true=self.y_test, y_pred=y_pred))

        writer = Writer()
        writer.newDataCsv(header=self.header,
                          X_samp_new=np.concatenate((X_train, X_pred), axis=0),
                          y_samp_new=np.concatenate((y_train, y_pred), axis=0))
        return y_pred

    def voting(self):
        _, _, _, y_test, y_pred1 = self.run('mlp')
        _, _, _, y_test, y_pred2 = self.run('nb')
        _, _, _, y_test, y_pred3 = self.run('neater')
        _, _, _, y_test, y_pred4 = self.run('rf')
        _, _, _, y_test, y_pred5 = self.run('svm')
        _, _, _, y_test, y_pred6 = self.run('knn')

        mode, count = st.mode([y_pred1, y_pred2, y_pred3, y_pred6])
        y_pred = mode[0]
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report

        print("with Voting:")
        print(confusion_matrix(y_true=y_test, y_pred=y_pred))
        print(classification_report(y_true=y_test, y_pred=y_pred))
        return y_pred

    def run(self, method):
        return self.X_train, self.y_train, self.X_test, self.y_test, getattr(self, method)()
