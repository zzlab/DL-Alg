import minpy.numpy as np
import minpy.nn.model_builder as builder

from minpy.context import set_context, gpu
set_context(gpu(0))

import cPickle as pickle

import sys
sys.path.append('../../nn')
from custom_layers import *
from solver_primitives import *

sys.path.append('../')
from utilities.data_utility import load_cifar10
# data = load_cifar10(path='../utilities/cifar/')
data = load_cifar10(path='../utilities/cifar/', center=True, rescale=True)

hidden_layers = 4
shapes = (1024,) * hidden_layers + (10,)
activation = builder.ReLU
storage = {}
mlp = builder.Sequential()
for i, shape in enumerate(shapes[:-1]):
  mlp.append(CenteredWeightAffine(shape, True))
  mlp.append(builder.Export('affine%d' % i, storage))
# mlp.append(builder.Affine(shape))
  mlp.append(activation())
mlp.append(CenteredWeightAffine(shapes[-1], True))
mlp.append(builder.Export('affine%d' % (len(shapes) - 1), storage))
# mlp.append(builder.Affine(shapes[-1]))
model = builder.Model(mlp, 'softmax', (3072,))

batch_size = 100
batches = len(data[0]) // batch_size
batch_index = 0

iterations = 1000
interval = 10
checkpoint_interval = 1
# settings = {}
settings = {'learning_rate' : 0.005}
initialize(model)
updater = Updater(model, 'sgd_momentum', settings)

for i in range(iterations):
  X_batch = data[0][batch_index * batch_size : (batch_index + 1) * batch_size]
  Y_batch = data[1][batch_index * batch_size : (batch_index + 1) * batch_size]
  batch_index = (batch_index + 1) % batches

  gradients, loss = gradient_loss(model, X_batch, Y_batch)
  loss = loss.asnumpy()[0]

  '''
  for g in gradients:
    print np.min(g), np.max(g)
  '''

  updater.update(gradients)

  if (i + 1) % interval == 0:
    print 'iteration %d loss %f' % (i + 1, loss)

  if (i + 1) % checkpoint_interval == 0:
    file_name = 'centered-weight-checkpoints/iteration-%d' % (i + 1)
    for key, value in storage.items():
      storage[key] = value.asnumpy()
    pickle.dump(storage, open(file_name, 'wb'))
    print 'iteration %d checkpointed' % (i + 1)
