from utils.NEATER import NEATER
from utils.Reader import Reader
import matplotlib.pyplot as plt
import imbalanced_databases as imbd
from utils.Poly import Poly
import numpy as np
dataset= imbd.load_glass0()
X, y= dataset['data'], dataset['target']

r = Reader('normalized_noise_vectors.csv')
X , y , noises = r.construct()
print(X[y == 0][:,15])
print(X[y == 1][:,16])
print(len(X[y == 0][:,15] ))
print(len(X[y == 1][:,16] ))
plt.figure(figsize=(10, 5))
plt.scatter(X[y == 0][:,15], X[y == 0][:,16], label='majority class', c='orange')
plt.scatter(X[y == 1][:,15], X[y == 1][:,16], label='minority class', c='olive')
plt.title('original dataset')
plt.xlabel('coordinate 0')
plt.ylabel('coordinate 1')
plt.legend()


print('majority class: %d' % np.sum(y == 0))
print('minority class: %d' % np.sum(y == 1))

p = Poly()

X_samp, y_samp= p.sample(X = X, y = y)


print('majority class: %d' % np.sum(y_samp == 0))
print('minority class: %d' % np.sum(y_samp == 1))

X_samp, y_samp= X_samp[len(X):], y_samp[len(y):]

# printing the number of new samples
print('majority new samples: %d' % np.sum(y_samp == 0))
print('minority new samples: %d' % np.sum(y_samp == 1))


plt.figure(figsize=(10, 5))

plt.scatter(X[y == 0][:,15], X[y == 0][:,16], label='majority class', c='orange', marker='o')
plt.scatter(X[y == 1][:,15], X[y == 1][:,16], label='minority class', c='olive', marker='o')
plt.scatter(X_samp[y_samp == 1][:,15], X_samp[y_samp == 1][:,16], label='new minority samples', c='olive', marker='x')
plt.title('oversampled dataset')
plt.xlabel('coordinate 0')
plt.ylabel('coordinate 1')
plt.show()


