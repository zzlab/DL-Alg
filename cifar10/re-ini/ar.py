import minpy.numpy as np
import minpy.nn.model_builder as builder

from minpy.context import set_context, cpu, gpu

import cPickle as pickle

import sys
sys.path.append('../../nn')
from custom_layers import *
from facility import *
from solver_primitives import *

sys.path.append('../')
from utilities.data_utility import load_cifar10
data = load_cifar10(path='../utilities/cifar/', center=True, rescale=True)

'''
print sys.argv
raise Exception()
'''
# ACTIVATION = sys.argv[1]
ACTIVATION = 'ReLU'
activation = getattr(builder, ACTIVATION)
# DR_INTERVAL = int(sys.argv[2])
DR_INTERVAL = 30
# DEVICE = int(sys.argv[3])
DEVICE = 0
set_context(gpu(DEVICE))
# shapes = [int(shape) for shape in sys.argv[4:]]
shapes = (1024,) * 10 + (10,)

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

iterations = 25000
interval = 10

# settings = {}
settings = {'learning_rate' : 0.005}
initialize(model)
updater = Updater(model, 'sgd', settings)

for key, value in model.params.items():
  print key, value.context

loss_history = []

for i in range(iterations):
  X_batch = data[0][batch_index * batch_size : (batch_index + 1) * batch_size]
  Y_batch = data[1][batch_index * batch_size : (batch_index + 1) * batch_size]
  batch_index = (batch_index + 1) % batches

  gradients, loss = gradient_loss(model, X_batch, Y_batch)
  loss = loss.asnumpy()[0]
  loss_history.append(loss)

  '''
  minimum = min(np.max(np.abs(g)) for g in gradients)
  print minimum
  '''

  print 'parameters'
  for key, value in model.params.items():
    if 'weight' in key:
      print key, np.mean(value ** 2)

  updater.update(gradients)

  if (i + 1) % DR_INTERVAL == 0:
    affine_rescale(mlp, X_batch, model.params)

  if (i + 1) % interval == 0:
    print 'iteration %d loss %f' % (i + 1, loss)

'''
configuration = 'interval-%s-shape-%s' % (DR_INTERVAL, '-'.join(sys.argv[4:]))
pickle.dump(loss_history, open('mlp-dr-loss-%s' % configuration, 'wb'))
'''
