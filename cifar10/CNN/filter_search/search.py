import sys
sys.path.append('../')
sys.path.append('../../nn')

import utilities.data_utility as data_utility
import utilities.GPU_utility as GPU_utility

import MXModels as Models
import MXInitializer as Initializer
import MXSolver as Solver

import cPickle as pickle

class PooledCNN:
  def __init__(self, activation, filter_count, initializer):
    self.activation   = activation
    self.filter_count = filter_count
    self.initializer  = initializer
  def __call__(self, data_shape):
    import MXLayers as layers
    def __convolution(*args, **kwargs):
      '''
      if self.activation == 'NDReLU':
        return layers.NDReLUConvolution(*args, data_shape=data_shape, **kwargs)
      '''
      return layers.convolution(*args, activation=self.activation, data_shape=data_shape, **kwargs)
    inputs = layers.variable('data')
    outputs = __convolution(inputs, (3, 3), self.filter_count, pad=(1, 1))
    outputs = __convolution(outputs, (1, 1), self.filter_count)
    outputs = __convolution(outputs, (1, 1), self.filter_count)
    outputs = layers.pooling(outputs, 'max', (2, 2), (2, 2))
    outputs = __convolution(outputs, (1, 1), self.filter_count)
    outputs = __convolution(outputs, (1, 1), self.filter_count)
    outputs = layers.pooling(outputs, 'max', (2, 2), (2, 2))
    outputs = __convolution(outputs, (1, 1), self.filter_count)
    outputs = __convolution(outputs, (1, 1), self.filter_count)
    outputs = __convolution(outputs, (8, 8), 10, stride=(8, 8))
    outputs = layers.flatten(outputs)
    outputs = layers.softmax(outputs)
    return outputs, self.initializer 

activation = sys.argv[1]
filter_count = int(sys.argv[2])
model = PooledCNN(activation, filter_count, Initializer.DReLUInitializer(2.0))

optimizer_settings = {
  'lr'                : 0.01,
  'lr_decay_interval' : 1,
  'lr_decay_factor'   : 1.0,
  'optimizer'         : 'SGD',
  'weight_decay'      : 0
}

data = data_utility.load_cifar10(path='../utilities/cifar/', reshape=True, center=True, rescale=True)

solver_configuration = {
  'batch_size'         : 128,
  'data'               : data,
  'devices'            : GPU_utility.GPU_availability()[:3],
  'epoch'              : 60,
  'optimizer_settings' : optimizer_settings,
}

label = 'pooled-CNN-%s-%d-filters-lr-%f' % (activation, filter_count, optimizer_settings['lr'])

solver = Solver.MXSolver(model, **solver_configuration)
history = solver.train()
pickle.dump(history, open('../../models/%s-history' % label, 'wb'))
pickle.dump((solver.model.arg_params, solver.model.aux_params), open('../../models/%s-parameters' % label, 'wb'))
