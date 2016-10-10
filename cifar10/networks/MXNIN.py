import sys
sys.path.append('../')
sys.path.append('../../nn')

import cPickle as pickle

from utilities.data_utility import load_cifar10
from utilities.data_utility import load_whitened_cifar10
from utilities.GPU_utility import GPU_availability
from MXModels.NIN import NIN
from MXSolver import *
from MXInitializer import *

optimizer_settings = {
  'lr'                : 0.1,
  'momentum'          : 0.9,
  'optimizer'         : 'SGD',
  'scheduler'         : 'mannual',
  'weight_decay'      : 0.0001
}

print 'loading data'
data = load_whitened_cifar10(path='../utilities/whitened-cifar/', reshape=True)
# data = load_cifar10(path='../utilities/cifar/', reshape=True, center=True, rescale=True)
print 'data loaded'

solver_configuration = {
  'batch_size'         : 128,
  'data'               : data,
  'devices'            : GPU_availability()[:4],
  'epoch'              : 300,
  'file'               : '../../nn/lr',
  'optimizer_settings' : optimizer_settings,
  'verbose'            : True
}

activation = 'DReLU'
model = NIN(activation, DReLUInitializer(magnitude=2.0))
solver = MXSolver(model, **solver_configuration)
history = solver.train() # test_accuracy, progress
path = 'NIN-%s' % activation
pickle.dump(history, open('../models/%s' % path, 'wb'))
