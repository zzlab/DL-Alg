import numpy as np
from numpy.dual import svd

from data_utility import load_cifar10
X, Y, validation_X, validation_Y, test_X, test_Y = load_cifar10(center=True)

print 'computing zca matrix'
covariance = np.dot(X.T, X) / len(X)
U, S, V = svd(covariance)
epsilon = 1E-4
zca_matrix = np.dot(np.dot(U, np.diag(1.0 / (S + epsilon) ** 0.5)), U.T)

print 'whitening'
X = np.dot(X, zca_matrix)
X = X.astype(np.float32)
validation_X = np.dot(validation_X, zca_matrix)
validation_X = validation_X.astype(np.float32)
test_X = np.dot(test_X, zca_matrix)
test_X = test_X.astype(np.float32)

import cPickle as pickle
print 'dumping'
path = 'whitened-cifar'
BATCH_SIZE = 10000
for i in range(len(X) / BATCH_SIZE):
  pickle.dump(
    {
      'data' : X[i * BATCH_SIZE : (i + 1) * BATCH_SIZE],
      'labels' : Y[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    },
    open('%s/training%d' % (path, i), 'wb')
  )
pickle.dump(
  {'data' : validation_X, 'labels' : validation_Y},
  open('%s/validation' % path, 'wb')
)
pickle.dump(
  {'data' : test_X, 'labels' : test_Y},
  open('%s/test' % path, 'wb')
)
