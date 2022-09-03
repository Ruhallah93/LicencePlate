from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
from utils.NEATER import NEATER
from utils.Reader import Reader
from utils.Writer import Writer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
class Test:
    def __init__(self):
        pass

    def test_data(self, test_size, random_state, test_file):
        self.test_size = test_size
        self.random_state = random_state
        self.test_file = test_file
        reader = Reader(test_file)
        X, y, header = reader.construct()
        print('0 class in test: %d' % np.sum(y == 0))
        print('1 class in test: %d' % np.sum(y == 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        print("with MLP:")
        clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(confusion_matrix(y_test, y_pred))

        print("with SVC:")
        clf =  SVC(gamma='auto').fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        print(confusion_matrix(y_test, y_pred))

        print("with NB:")
        gnb = GaussianNB().fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        print(confusion_matrix(y_test, y_pred))

        print("with RF:")
        clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(confusion_matrix(y_test, y_pred))

        print('0 class in test: %d' % np.sum(y_test == 0))
        print('1 class in test: %d' % np.sum(y_test == 1))
        print('0 class in train: %d' % np.sum(y_train == 0))
        print('1 class in train: %d' % np.sum(y_train == 1))

        # X_train1 = np.concatenate((X_train , X_train[y_train == 0] ) , axis = 0)
        # y_train1 = np.concatenate((y_train, y_train[y_train == 0]), axis=0)
        # X_train = np.concatenate((X_train , X_train1[y_train1 == 0] ) , axis = 0)
        # y_train = np.concatenate((y_train, y_train1[y_train1 == 0]), axis=0)

        oversample = RandomOverSampler(sampling_strategy= 0.9)
        # fit and apply the transform
        X_train, y_train = oversample.fit_resample(X_train, y_train)

        print('0 class in train oversampled: %d' % np.sum(y_train == 0))
        print('1 class in train oversampled: %d' % np.sum(y_train == 1))

        n = NEATER()
        X_pred, y_pred = n.sample(X_train, y_train, X_test, y_test, data_produced=0)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        writer = Writer()
        writer.newDataCsv(header=header, X_samp_new=np.concatenate((X_train, X_pred), axis=0),
                          y_samp_new=np.concatenate((y_train, y_pred), axis=0))

        return X_train, y_train, X_pred, y_pred
