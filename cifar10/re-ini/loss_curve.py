import minpy.numpy as np
import minpy.nn.model_builder as builder

from minpy.context import set_context, cpu, gpu
# set_context(gpu(0))
set_context(cpu())

import cPickle as pickle

import sys
sys.path.append('../../nn')
from custom_layers import *
from facility import *
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
  mlp.append(activation())
mlp.append(builder.Affine(shapes[-1]))
model = builder.Model(mlp, 'softmax', (3072,))

batch_size = 100
batches = len(data[0]) // batch_size
batch_index = 0
# raise Exception()

iterations = 20000
interval = 10
rescaling_interval = 1000

# settings = {}
settings = {'learning_rate' : 0.01}
initialize(model)
updater = Updater(model, 'sgd', settings)

loss_history = []
mean = {key : [] for key in model.params}
std = {key : [] for key in model.params}
L_2 = {key : [] for key in model.params}
minimum = {key : [] for key in model.params}
maximum = {key : [] for key in model.params}

for i in range(iterations):
  X_batch = data[0][batch_index * batch_size : (batch_index + 1) * batch_size]
  Y_batch = data[1][batch_index * batch_size : (batch_index + 1) * batch_size]
  batch_index = (batch_index + 1) % batches

  gradients, loss = gradient_loss(model, X_batch, Y_batch)
  loss = loss.asnumpy()[0]
  loss_history.append(loss)

  for key, value in zip(model.params.keys(), gradients):
    mean[key].append(np.mean(value).asnumpy())
    std[key].append(np.std(value).asnumpy())
    L_2[key].append(np.mean(value ** 2).asnumpy())
    minimum[key].append(np.min(value).asnumpy())
    maximum[key].append(np.max(value).asnumpy())

  updater.update(gradients)

  if (i + 1) % rescaling_interval == 0:
    rescale(mlp, data[2], model.params) # validation data
    print 'rescaled'

  if (i + 1) % interval == 0:
    print 'iteration %d loss %f' % (i + 1, loss)

pickle.dump(
  (loss_history, mean, std, L_2, minimum, maximum),
  open('dr-g-norm-%d' % rescaling_interval, 'wb')
)
