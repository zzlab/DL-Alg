import minpy.numpy as np
import minpy.nn.model_builder as builder

from minpy.context import set_context, cpu, gpu
set_context(gpu(3))
# set_context(cpu())

import cPickle as pickle

import sys
sys.path.append('../../nn')
from custom_layers import *
from facility import *
from solver_primitives import *

sys.path.append('../')
from utilities.data_utility import load_cifar10
data = load_cifar10(path='../utilities/cifar/', center=True, rescale=True)

activation = builder.ReLU
shapes = (3072,) * 4 + (10,)
storage = {}
mlp = builder.Sequential()
for i, shape in enumerate(shapes[:-1]):
  mlp.append(builder.Affine(shape))
  mlp.append(activation())
mlp.append(builder.Affine(shapes[-1]))
model = builder.Model(mlp, 'softmax', (3072,))

batch_size = 100
batches = len(data[0]) // batch_size
batch_index = 0

ITERATIONS = 25000
INTERVAL = 10
DR_INTERVAL = 4

# settings = {}
settings = {'learning_rate' : 0.05}
initialize(model)
updater = Updater(model, 'sgd', settings)

loss_history = []

for i in range(ITERATIONS):
  X_batch = data[0][batch_index * batch_size : (batch_index + 1) * batch_size]
  Y_batch = data[1][batch_index * batch_size : (batch_index + 1) * batch_size]
  batch_index = (batch_index + 1) % batches

  gradients, loss = gradient_loss(model, X_batch, Y_batch)
  loss = loss.asnumpy()[0]
  loss_history.append(loss)

  updater.update(gradients)

  if (i + 1) % DR_INTERVAL == 0:
    outputs, factors = rescale(mlp, X_batch, model.params)
    '''
    outputs = model.forward(X_batch, 'train')
    rescaled_loss = model.loss(outputs, Y_batch)
    print 'loss', loss, 'rescaled', rescaled_loss
    '''

  if (i + 1) % INTERVAL == 0:
    print 'iteration %d loss %f' % (i + 1, loss)
