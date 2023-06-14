import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from synthetic_plates.ImageFiltering.utils._metric_tensor import (NearestNeighborsWithMetricTensor , MetricLearningMixin)
from synthetic_plates.ImageFiltering.utils.NEATERJUNIOR import NEATERJUNIOR
from synthetic_plates.ImageFiltering.utils.NoRegretJunior import NoRegretJunior
from synthetic_plates.ImageFiltering.utils.CumulativePayoffJunior import CumulativePayoffJunior
from sklearn.neighbors import DistanceMetric

train_data = pd.read_csv('train_data/double_label/example_data.csv')
features = train_data.columns.drop(['instance_name', 'label', 'label_noise', 'label_rotation'])
rotation_features = ['scale' , 'roll']

X_rotation = train_data[rotation_features].to_numpy()[:12][:]
y_rotation = train_data['label_rotation'].replace(['Correct', 'Corrupt'], [1, 0]).to_numpy()[:12][:]

X_rotation_train, X_rotation_test,  y_rotation_train , y_rotation_test =train_test_split(X_rotation, y_rotation, test_size=0.5, shuffle=True)

labels = np.zeros(len(y_rotation))
for i in range(0 ,len(y_rotation)):
    labels[i] = y_rotation[i]

#X_TSNE = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=12).fit_transform(X_rotation)

# X_test = np.zeros((10,2))
# X_train = np.zeros((10,2))
#
# for i in range(0 , 10):
#     ind1 = np.where(np.all(X_rotation == X_rotation_test[i], axis=1))[0]
#     if len(ind1) > 0:
#         X_test[i] = X_TSNE[ind1]
#     ind2 = np.where(np.all(X_rotation == X_rotation_train[i], axis=1))[0]
#     if len(ind2) > 0:
#         X_train[i] = X_TSNE[ind2]




nn = NearestNeighborsWithMetricTensor(n_neighbors=5,
                                      n_jobs=1,
                                      metric="euclidean")

nn.fit(X_rotation)
indices = nn.kneighbors(X_rotation_test, 5, return_distance=False)
dist = DistanceMetric.get_metric(metric="euclidean")
dm = dist.pairwise(X_rotation_test, X_rotation)
dm[dm == 0] = 1e-8
dm = 1/dm
syn = [row[0] for row in indices]

labels[syn] = 0.5

for k in range(0 , 5):
    plt.figure(figsize=(10, 5))
    plt.scatter(X_rotation_train[np.where(y_rotation_train == 0)][:,0],X_rotation_train[np.where(y_rotation_train == 0)][:,1],
                            label='corrupt',
                            c='red', s = 100 )
    plt.scatter(X_rotation_train[np.where(y_rotation_train == 1)][:,0],X_rotation_train[np.where(y_rotation_train == 1)][:,1],
                            label='correct',
                            c='blue', s = 100)

    plt.scatter(X_rotation_test[:,0], X_rotation_test[:,1],
                            label='unlabeled',
                            c='black', marker="*" ,s = 100)

    r, g, b = 30, 144, 255
    neon_blue = (r/255, g/255, b/255)

    for i in range(0 , len(X_rotation)):
        if labels[i] == 1 and i not in syn:
            plt.text(X_rotation[i][0] + 0.02, X_rotation[i][1] + 0.02, round(labels[i],3) , color= neon_blue)
        elif labels[i] == 0 and i not in syn:
            plt.text(X_rotation[i][0] + 0.02, X_rotation[i][1] + 0.02, round(labels[i],3), color="red")
        else:
            plt.text(X_rotation[i][0] + 0.02, X_rotation[i][1] + 0.02, round(labels[i],3) , color= "black")
            plt.text(X_rotation[i][0] - 0.02, X_rotation[i][1] - 0.02, y_rotation[i], color="orange")

    for i in range(0,len(X_rotation_test)):
        for j in range(1 , 5):
            plt.plot([X_rotation[indices[i][0]][0] , X_rotation[indices[i][j]][0]] , [X_rotation[indices[i][0]][1] , X_rotation[indices[i][j]][1]] , color= 'black',alpha=0.5)


    plt.title('Iteration ' + str(k + 1) + ' of NEATER')
    plt.show()
    neaterJR = NEATERJUNIOR(X_rotation_test , indices , dm , labels, 0.999999)
    labels = neaterJR.probabilityOfCorrect()


labels = np.zeros(len(y_rotation))
for i in range(0 ,len(y_rotation)):
    labels[i] = y_rotation[i]
labels[syn] = 0.5

cumulativeRegretCorrupt = np.zeros(10)
cumulativeRegretCorrect = np.zeros(10)
for k in range(0 , 5):
    plt.figure(figsize=(10, 5))
    plt.scatter(X_rotation_train[np.where(y_rotation_train == 0)][:,0],X_rotation_train[np.where(y_rotation_train == 0)][:,1],
                            label='corrupt',
                            c='red', s = 100 )
    plt.scatter(X_rotation_train[np.where(y_rotation_train == 1)][:,0],X_rotation_train[np.where(y_rotation_train == 1)][:,1],
                            label='correct',
                            c='blue', s = 100)

    plt.scatter(X_rotation_test[:,0], X_rotation_test[:,1],
                            label='unlabeled',
                            c='black', marker="*" ,s = 100)

    r, g, b = 30, 144, 255
    neon_blue = (r/255, g/255, b/255)

    for i in range(0 , len(X_rotation)):
        if labels[i] == 1 and i not in syn:
            plt.text(X_rotation[i][0] + 0.02, X_rotation[i][1] + 0.02, round(labels[i],3) , color= neon_blue)
        elif labels[i] == 0 and i not in syn:
            plt.text(X_rotation[i][0] + 0.02, X_rotation[i][1] + 0.02, round(labels[i],3), color="red")
        else:
            plt.text(X_rotation[i][0] + 0.02, X_rotation[i][1] + 0.02, round(labels[i],3) , color= "black")
            plt.text(X_rotation[i][0] - 0.02, X_rotation[i][1] - 0.02, y_rotation[i], color="orange")

    for i in range(0,len(X_rotation_test)):
        for j in range(1 , 5):
            plt.plot([X_rotation[indices[i][0]][0] , X_rotation[indices[i][j]][0]] , [X_rotation[indices[i][0]][1] , X_rotation[indices[i][j]][1]] , color= 'black',alpha=0.5)


    plt.title('Iteration ' + str(k + 1) + ' of Regret')
    plt.show()
    regretJR = NoRegretJunior(X_rotation_test , indices , dm , labels, cumulativeRegretCorrect , cumulativeRegretCorrupt)
    labels , cumulativeRegretCorrect , cumulativeRegretCorrupt = regretJR.probabilityOfCorrect()

labels = np.zeros(len(y_rotation))
for i in range(0 ,len(y_rotation)):
    labels[i] = y_rotation[i]
labels[syn] = 0.5

cumulativePayoffCorrupt = np.zeros(10)
cumulativePayoffCorrect = np.zeros(10)
for k in range(0 , 5):
    plt.figure(figsize=(10, 5))
    plt.scatter(X_rotation_train[np.where(y_rotation_train == 0)][:,0],X_rotation_train[np.where(y_rotation_train == 0)][:,1],
                            label='corrupt',
                            c='red', s = 100 )
    plt.scatter(X_rotation_train[np.where(y_rotation_train == 1)][:,0],X_rotation_train[np.where(y_rotation_train == 1)][:,1],
                            label='correct',
                            c='blue', s = 100)

    plt.scatter(X_rotation_test[:,0], X_rotation_test[:,1],
                            label='unlabeled',
                            c='black', marker="*" ,s = 100)

    r, g, b = 30, 144, 255
    neon_blue = (r/255, g/255, b/255)

    for i in range(0 , len(X_rotation)):
        if labels[i] == 1 and i not in syn:
            plt.text(X_rotation[i][0] + 0.02, X_rotation[i][1] + 0.02, round(labels[i],3) , color= neon_blue)
        elif labels[i] == 0 and i not in syn:
            plt.text(X_rotation[i][0] + 0.02, X_rotation[i][1] + 0.02, round(labels[i],3), color="red")
        else:
            plt.text(X_rotation[i][0] + 0.02, X_rotation[i][1] + 0.02, round(labels[i],3) , color= "black")
            plt.text(X_rotation[i][0] - 0.02, X_rotation[i][1] - 0.02, y_rotation[i], color="orange")

    for i in range(0,len(X_rotation_test)):
        for j in range(1 , 5):
            plt.plot([X_rotation[indices[i][0]][0] , X_rotation[indices[i][j]][0]] , [X_rotation[indices[i][0]][1] , X_rotation[indices[i][j]][1]] , color= 'black',alpha=0.5)


    plt.title('Iteration ' + str(k + 1) + ' of Cumulative Payoff')
    plt.show()
    cpJR = CumulativePayoffJunior(X_rotation_test , indices , dm , labels, 0.5,  cumulativePayoffCorrect , cumulativePayoffCorrupt)
    labels , cumulativePayoffCorrect , cumulativePayoffCorrupt = cpJR.probabilityOfCorrect()

