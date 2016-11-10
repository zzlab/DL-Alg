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

from loss import stochastic_gradient_loss

sys.path.append('../')
from utilities.data_utility import load_mnist
data = load_mnist(path='../utilities')

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
model = builder.Model(mlp, 'softmax', (28 * 28,))

batch_size = 50
batches = len(data[0]) // batch_size
batch_index = 0

iterations = 10000
interval = 10
validation_interval = 1000
validation_X, validation_Y = data[2 : 4]
validation_X = validation_X[:1024]
validation_Y = validation_Y[:1024]

settings = {'learning_rate' : 0.01}
initialize(model)
updater = Updater(model, 'sgd', settings)

for i in range(iterations):
  X_batch = data[0][batch_index * batch_size : (batch_index + 1) * batch_size]
  Y_batch = data[1][batch_index * batch_size : (batch_index + 1) * batch_size]
  batch_index = (batch_index + 1) % batches

  gradients, loss = stochastic_gradient_loss(model, X_batch, Y_batch, 0.000)
  loss = loss.asnumpy()[0]

  updater.update(gradients)

  if (i + 1) % interval == 0:
    print 'iteration %d loss %f' % (i + 1, loss)
  if (i + 1) % validation_interval == 0:
    outputs = model.forward(validation_X, 'train')
    predicted_Y = np.argmax(outputs, axis=1)
    errors = np.count_nonzero(predicted_Y - validation_Y)
    accuracy = 1 - errors / float(validation_Y.shape[0])
    print accuracy
