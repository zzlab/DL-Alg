import minpy.numpy as np
import minpy.nn.model_builder as builder

from minpy.context import set_context, cpu, gpu
# set_context(gpu(0))
set_context(cpu())

import cPickle as pickle

import sys
sys.path.append('../../nn')
from custom_layers import *
from solver_primitives import *

sys.path.append('../')
from utilities.data_utility import load_cifar10
# data = load_cifar10(path='../utilities/cifar/')
print 'loading data'
data = load_cifar10(path='../utilities/cifar/', center=True, rescale=True)
print 'data loaded'

hidden_layers = 4
shapes = (1024,) * hidden_layers + (10,)
activation = builder.ReLU
storage = {}
mlp = builder.Sequential()
for i, shape in enumerate(shapes[:-1]):
  mlp.append(builder.Affine(shape))
  mlp.append(builder.Export('affine%d' % i, storage))
  mlp.append(activation())
mlp.append(builder.Affine(shapes[-1]))
mlp.append(builder.Export('affine%d' % (len(shapes) - 1), storage))
model = builder.Model(mlp, 'softmax', (3072,))
print 'model created'

batch_size = 100
batches = len(data[0]) // batch_size
batch_index = 0
# raise Exception()

iterations = 10000
interval = 10
checkpoint_interval = 500
validation_interval = 10
validation_X, validation_Y = data[2 : 4]

# settings = {}
settings = {'learning_rate' : 0.01}
initialize(model)
print 'model initialized'
updater = Updater(model, 'sgd_momentum', settings)
print 'updater initialized'

for i in range(iterations):
  X_batch = data[0][batch_index * batch_size : (batch_index + 1) * batch_size]
  Y_batch = data[1][batch_index * batch_size : (batch_index + 1) * batch_size]
  batch_index = (batch_index + 1) % batches

  gradients, loss = gradient_loss(model, X_batch, Y_batch)
  loss = loss.asnumpy()[0]

  updater.update(gradients)

  if (i + 1) % interval == 0:
    print 'iteration %d loss %f' % (i + 1, loss)

  if (i + 1) % checkpoint_interval == 0:
    file_name = 'training-parameters-checkpoints/parameters-iteration-%d' % (i + 1)
    to_dump = { key : value.asnumpy() for key, value in model.params.items() }
    pickle.dump(to_dump, open(file_name, 'wb'))
    print 'iteration %d checkpointed' % (i + 1)
