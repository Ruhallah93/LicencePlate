from utils.NEATER import NEATER
from utils.Reader import Reader
from utils.Writer import Writer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from utils.Test import Test

# r = Reader('normalized-noise-vectors.csv')
# X , y , noises = r.construct()

t = Test()
X, y, X_samp, y_samp = t.test_data(0.1, 42, 'normalized-noise-vectors.csv')

X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(X)

plt.figure(figsize=(10, 5))
plt.scatter(X_embedded[y == 0][:, 0], X_embedded[y == 0][:, 1], label='majority class', c='orange')
plt.scatter(X_embedded[y == 1][:, 0], X_embedded[y == 1][:, 1], label='minority class', c='olive')
plt.title('original dataset')
plt.xlabel('coordinate 0')
plt.ylabel('coordinate 1')
plt.legend()

print('majority class: %d' % np.sum(y == 0))
print('minority class: %d' % np.sum(y == 1))

# n = NEATER()
# X_samp, y_samp= n.sample(X = X, y = y , X_test = X, y_test = y, data_produced= 1)


print('majority class: %d' % np.sum(y_samp == 0))
print('minority class: %d' % np.sum(y_samp == 1))

# X_samp, y_samp= X_samp[len(X):], y_samp[len(y):]


# w = Writer()
# w.newDataCsv(X_samp , y_samp , noises)


# printing the number of new samples
print('majority new samples: %d' % np.sum(y_samp == 0))
print('minority new samples: %d' % np.sum(y_samp == 1))

Xs_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(X_samp)

plt.figure(figsize=(10, 5))

plt.scatter(X_embedded[y == 0][:, 0], X_embedded[y == 0][:, 1], label='majority class', c='orange', marker='o')
plt.scatter(X_embedded[y == 1][:, 0], X_embedded[y == 1][:, 1], label='minority class', c='olive', marker='o')
plt.scatter(Xs_embedded[y_samp == 1][:, 0], Xs_embedded[y_samp == 1][:, 1], label='new minority samples', c='olive',
            marker='x')
plt.scatter(Xs_embedded[y_samp == 0][:, 0], Xs_embedded[y_samp == 0][:, 1], label='new minority samples', c='red',
            marker='x')
plt.title('oversampled dataset')
plt.xlabel('coordinate 0')
plt.ylabel('coordinate 1')
plt.show()
