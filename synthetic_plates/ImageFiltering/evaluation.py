import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from utils.Evaluation import Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import csv
import os

# r = Reader('normalized_noise_vectors.csv')
# X , y , noises = r.construct()

visualization = False
test_size = 0.3
# ['knn', 'mlp', 'nb', 'neater', 'voting', 'svm', 'rf']
for method in ['neater']:
    precisions = np.empty((0, 2), float)
    recalls = np.empty((0, 2), float)
    accuracies = []
    f1s = []
    for i in range(100):
        print("Method:", method)
        evaluation = Evaluation(test_size, 'train_data/double_label/normalized_noise_vectors.csv')
        X_train, y_train, X_test, y_test, y_test_noise, y_test_rotation, y_pred_noises, y_pred_rotations = evaluation.run(
            method=method)

        y_pred = y_pred_noises * y_pred_rotations

        print(confusion_matrix(y_true=y_test_noise, y_pred=y_pred))
        report = classification_report(y_true=y_test_rotation, y_pred=y_pred)
        print(report)

        a = precision_score(y_true=y_test_noise, y_pred=y_pred_noises, average=None) + precision_score(
            y_true=y_test_rotation, y_pred=y_pred_rotations, average=None)
        precisions = np.append(precisions, [a / 2], axis=0)
        a = recall_score(y_true=y_test_noise, y_pred=y_pred_noises, average=None) + recall_score(
            y_true=y_test_rotation,
            y_pred=y_pred_rotations,
            average=None)
        recalls = np.append(recalls, [a / 2], axis=0)
        a = accuracy_score(y_true=y_test_noise, y_pred=y_pred_noises) + accuracy_score(y_true=y_test_rotation,
                                                                                       y_pred=y_pred_rotations)
        accuracies = np.append(accuracies, a / 2)
        a = f1_score(y_true=y_test_noise, y_pred=y_pred_noises, average='macro') + f1_score(y_true=y_test_rotation,
                                                                                            y_pred=y_pred_rotations,
                                                                                            average='macro')
        f1s = np.append(f1s, a / 2)

        # printing the number of new samples
        print('0 class test samples: %d' % np.sum(y_test == 0))
        print('1 class test samples: %d' % np.sum(y_test == 1))

        X_all = np.concatenate([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])

        if visualization:
            X_em = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(X_all)
            plt.figure(figsize=(10, 5))
            plt.scatter(X_em[:len(X_train)][y_train == 0][:, 0], X_em[:len(X_train)][y_train == 0][:, 1],
                        label='Train: majority',
                        c='red', edgecolors='lime')
            plt.scatter(X_em[:len(X_train)][y_train == 1][:, 0], X_em[:len(X_train)][y_train == 1][:, 1],
                        label='Train: minority',
                        c='blue', edgecolors='lime')
            plt.scatter(X_em[len(X_train):][y_test == 0][:, 0], X_em[len(X_train):][y_test == 0][:, 1],
                        label='Test: majority',
                        facecolors='none', edgecolors='red')
            plt.scatter(X_em[len(X_train):][y_test == 1][:, 0], X_em[len(X_train):][y_test == 1][:, 1],
                        label='Test: minority',
                        facecolors='none', edgecolors='blue')
            plt.scatter(X_em[len(X_train):][y_pred == 0][:, 0], X_em[len(X_train):][y_pred == 0][:, 1],
                        label='Prediction: minority', marker='*', c='red')
            plt.scatter(X_em[len(X_train):][y_pred == 1][:, 0], X_em[len(X_train):][y_pred == 1][:, 1],
                        label='Prediction: minority', marker='*', c='blue')
            plt.legend()
            plt.title('oversampled dataset')
            plt.xlabel('coordinate 0')
            plt.ylabel('coordinate 1')
            plt.show()
    measures = {
        'method': method,
        'test_size': test_size,
        'metric': 'euclidean',
        'accuracy(m)': np.mean(accuracies),
        'accuracy(v)': np.std(accuracies),
        'f1(m)': np.mean(f1s),
        'f1(v)': np.std(f1s),
        'precision0(m)': np.mean(precisions[:, 0]),
        'precision0(v)': np.var(precisions[:, 0]),
        'precision1(m)': np.mean(precisions[:, 1]),
        'precision1(v)': np.var(precisions[:, 1]),
        'recall0(m)': np.mean(recalls[:, 0]),
        'recall0(v)': np.var(recalls[:, 0]),
        'recall1(m)': np.mean(recalls[:, 1]),
        'recall1(v)': np.var(recalls[:, 1]),
        '': ''
    }
    print(measures)
    file_name = 'r(' + str(test_size * 100)  + ').csv'
    file_exists = os.path.isfile(file_name)
    with open(file_name, 'a') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(measures.keys())
        writer.writerow(measures.values())
        csvfile.close()
