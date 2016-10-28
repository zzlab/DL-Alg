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
'''
data = (
  np.random.normal(0, 1, (40000, 3072)),
  np.random.choice(np.arange(10), 40000),
  np.random.normal(0, 1, (10000, 3072)),
  np.random.choice(np.arange(10), 10000),
  np.random.normal(0, 1, (10000, 3072)),
  np.random.choice(np.arange(10), 10000),
)
'''
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
batches = data[0].shape[0] // batch_size
batch_index = 0
# raise Exception()

iterations = 10000
interval = 10
checkpoint_interval = 100
validation_interval = 100
rescaling_interval = 1000
validation_X, validation_Y = data[2 : 4]

# settings = {}
settings = {'learning_rate' : 0.01}
initialize(model)
print 'model initialized'
updater = Updater(model, 'sgd_momentum', settings)
print 'updater initialized'

loss_history = []
for i in range(iterations):
  X_batch = data[0][batch_index * batch_size : (batch_index + 1) * batch_size]
  Y_batch = data[1][batch_index * batch_size : (batch_index + 1) * batch_size]
  batch_index = (batch_index + 1) % batches

  gradients, loss = gradient_loss(model, X_batch, Y_batch)
  loss = loss.asnumpy()[0]
  loss_history.append(loss)

  updater.update(gradients)

  if (i + 1) % interval == 0:
    print 'iteration %d loss %f' % (i + 1, loss)

  if (i + 1) % rescaling_interval == 0:
    rescale(mlp, data[2], model.params) # validation data
    print 'rescaled'

  '''
  if (i + 1) % checkpoint_interval == 0:
    # dump activation
    file_name = 'dynamic-rescaling/iteration-%d' % (i + 1)
    for key, value in storage.items():
      storage[key] = value.asnumpy()
    pickle.dump(storage, open(file_name, 'wb'))
    
    # dump parameters
    file_name = 'dynamic-rescaling/parameters-iteration-%d' % (i + 1)
    to_dump = { key : value.asnumpy() for key, value in model.params.items() }
    pickle.dump(to_dump, open(file_name, 'wb'))
    print 'iteration %d checkpointed' % (i + 1)
  '''

# end of iteration
pickle.dump(loss_history, open('dynamic-rescaling-loss-history', 'wb'))
