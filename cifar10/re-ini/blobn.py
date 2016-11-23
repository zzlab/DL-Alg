import minpy.numpy as np
import minpy.nn.model_builder as builder

import cPickle as pickle

import sys
sys.path.append('../../nn')
from custom_layers import *
from facility import *
from solver_primitives import *

sys.path.append('../')
from utilities.data_utility import load_cifar10
data = load_cifar10(path='../utilities/cifar/', center=True, rescale=True)

from minpy.context import set_context, cpu, gpu
print sys.argv
device = int(sys.argv[2])
set_context(gpu(device))

blob_setting = sys.argv[1]

shapes = (1024,) * 4 + (10,)
activation = ReLU
storage = {}
mlp = builder.Sequential()
for i, shape in enumerate(shapes[:-1]):
  mlp.append(builder.Affine(shape))
  mlp.append(BlobNormalization(blob_setting))
# mlp.append(builder.Export('bn%d' % i, storage))
  mlp.append(activation())
mlp.append(builder.Affine(shapes[-1]))
model = builder.Model(mlp, 'softmax', (3072,))

batch_size = 100
batches = len(data[0]) // batch_size
batch_index = 0

iterations = 25000
interval = 10

settings = {'learning_rate' : 0.05}
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
  loss = to_float(loss)
  loss_history.append(loss)

  '''
  for key, value in storage.items():
    print key, mean(value), std(value)
  '''

  updater.update(gradients)

  if (i + 1) % interval == 0:
    print 'iteration %d loss %f' % (i + 1, loss)

configuration = 'blob-%s-shape-%s' % (blob_setting, '-'.join(str(d) for d in shapes))
pickle.dump(loss_history, open('model/mlp-loss-%s' % configuration, 'wb'))
