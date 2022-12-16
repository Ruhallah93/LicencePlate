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
    def __init__(self, train_data_file, no_label_data_file):
        reader = Reader(train_data_file)
        self.X, self.y, self.header = reader.construct()
        reader = Reader(no_label_data_file)
        self.X_no_label, self.y_no_label, _ = reader.construct()

    def our(self):
        print('0 class in train oversampled: %d' % np.sum(self.y == 0))
        print('1 class in train oversampled: %d' % np.sum(self.y == 1))

        # balancing
        oversampling = RandomOverSampler(sampling_strategy=0.9)
        # fit and apply the transform
        x, y = oversampling.fit_resample(self.X, self.y)

        print('0 class in train oversampled: %d' % np.sum(y == 0))
        print('1 class in train oversampled: %d' % np.sum(y == 1))

        neater = NEATER()
        x_prediction, y_prediction = neater.sample(x, y, self.X_no_label, self.y_no_label, data_produced=0)

        writer = Writer()
        writer.newDataCsv(header=self.header,
                          X_samp_new=x_prediction,
                          y_samp_new=y_prediction)
        return y_prediction

    def run(self, method):
        return self.X_train, self.y_train, self.X_test, self.y_test, getattr(self, method)()
