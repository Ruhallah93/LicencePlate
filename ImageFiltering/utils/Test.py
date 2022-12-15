from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np

from ImageFiltering.utils.NEATER import NEATER
from ImageFiltering.utils.Reader import Reader
from ImageFiltering.utils.Writer import Writer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


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
        print("with MLP:")
        clf = MLPClassifier(random_state=1, max_iter=300).fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        return y_pred

    def svm(self):
        print("with SVC:")
        clf = SVC(gamma='auto').fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        return y_pred

    def nb(self):
        print("with NB:")
        gnb = GaussianNB().fit(self.X_train, self.y_train)
        y_pred = gnb.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        return y_pred

    def rf(self):
        print("with RF:")
        clf = RandomForestClassifier(max_depth=2, random_state=0).fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        return y_pred

    def our(self):
        # X_train1 = np.concatenate((X_train , X_train[y_train == 0] ) , axis = 0)
        # y_train1 = np.concatenate((y_train, y_train[y_train == 0]), axis=0)
        # X_train = np.concatenate((X_train , X_train1[y_train1 == 0] ) , axis = 0)
        # y_train = np.concatenate((y_train, y_train1[y_train1 == 0]), axis=0)

        oversample = RandomOverSampler(sampling_strategy=0.9)
        # fit and apply the transform
        X_train, y_train = oversample.fit_resample(self.X_train, self.y_train)

        print('0 class in train oversampled: %d' % np.sum(y_train == 0))
        print('1 class in train oversampled: %d' % np.sum(y_train == 1))

        n = NEATER()
        X_pred, y_pred = n.sample(X_train, y_train, self.X_test, self.y_test, data_produced=0)
        print(confusion_matrix(self.y_test, y_pred))
        print(classification_report(self.y_test, y_pred))

        writer = Writer()
        writer.newDataCsv(header=self.header,
                          X_samp_new=np.concatenate((X_train, X_pred), axis=0),
                          y_samp_new=np.concatenate((y_train, y_pred), axis=0))
        return y_pred

    def run(self, method):
        return self.X_train, self.y_train, self.X_test, self.y_test, getattr(self, method)()
