import sys
sys.path.append('../')
sys.path.append('../../nn')

import cPickle as pickle

from utilities.data_utility import load_cifar10
from utilities.GPU_utility import GPU_availability
from MXModels.NIN import NIN
from MXSolver import *
from MXInitializer import *

optimizer_settings = {
  'lr'                : 0.001,
  'lr_decay_interval' : 1,
  'lr_decay_factor'   : 1.0,
# 'momentum'          : 0,
  'optimizer'         : 'Adam',
  'weight_decay'      : 0
}

data = load_cifar10(path='../utilities/cifar/', reshape=True, center=True, rescale=True)
solver_configuration = {
  'batch_size'         : 64,
  'data'               : data,
  'devices'            : GPU_availability()[:4],
  'epoch'              : 300,
  'file'               : '../../nn/lr',
  'optimizer_settings' : optimizer_settings,
  'verbose'            : True
}

activation = 'DReLU'
model = NIN(activation, DReLUInitializer())
solver = MXSolver(model, **solver_configuration)
history = solver.train() # test_accuracy, progress
path = 'NIN-%s' % activation
pickle.dump(history, open('../models/%s' % path, 'wb'))
