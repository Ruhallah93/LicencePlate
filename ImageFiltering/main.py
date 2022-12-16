import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from utils.Evaluation import Test

# r = Reader('normalized-noise-vectors.csv')
# X , y , noises = r.construct()

t = Test(0.3, 'normalized-noise-vectors.csv')
# ['knn', 'mlp', 'nb', 'neater', 'voting', 'svm', 'rf']
for method in ['voting']:
    X_train, y_train, X_test, y_test, y_pred = t.run(method=method)

    # printing the number of new samples
    print('0 class test samples: %d' % np.sum(y_test == 0))
    print('1 class test samples: %d' % np.sum(y_test == 1))

    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    X_em = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(X_all)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_em[:len(X_train)][y_train == 0][:, 0], X_em[:len(X_train)][y_train == 0][:, 1],
                label='Train: majority',
                c='red', edgecolors='lime')
    plt.scatter(X_em[:len(X_train)][y_train == 1][:, 0], X_em[:len(X_train)][y_train == 1][:, 1],
                label='Train: minority',
                c='blue', edgecolors='lime')
    plt.scatter(X_em[len(X_train):][y_test == 0][:, 0], X_em[len(X_train):][y_test == 0][:, 1], label='Test: majority',
                facecolors='none', edgecolors='red')
    plt.scatter(X_em[len(X_train):][y_test == 1][:, 0], X_em[len(X_train):][y_test == 1][:, 1], label='Test: minority',
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
