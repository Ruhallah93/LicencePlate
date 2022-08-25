# import numpy as np
#
#
# def pairwise_distances_mahalanobis(X, Y=None, tensor=None):
#     if Y is None:
#         Y = X
#     if tensor is None:
#         tensor = np.eye(len(X[0]))
#         tensor[-2, i] = 1 / 6
#         tensor[-3, i] = 1 / 6
#         tensor[-4, i] = 1 / 6
#         for i in range(tensor.shape[0]):
#             if i == -2 or i == -3 or i == -4:
#                 tensor[i, i] = 1 / 6
#             else:
#                 tensor[i, i] = 1 / (len(X[0]) - 3)
#     tmp = (X[:, None] - Y)
#     power2 = tmp, np.dot(tmp, tensor)
#     power2 = np.einsum('ijk,ijk -> ij', tmp, np.dot(tmp, tensor))
#     return np.sqrt(.T)
#
#
#     dm = pairwise_distances_mahalanobis(X_all, X_syn, nn_params.get('metric_tensor', None))
