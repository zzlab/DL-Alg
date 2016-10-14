import sys
sys.path.append('../')
sys.path.append('../../nn')

import cPickle as pickle

from utilities.data_utility import load_cifar10
from utilities.data_utility import load_whitened_cifar10
from utilities.GPU_utility import GPU_availability
from LRScheduler import FactorScheduler
from MXModels.NIN import NIN
from MXSolver import *
from MXInitializer import *

initial_lr = 0.1
optimizer_settings = {
  'lr'                : initial_lr,
  'momentum'          : 0.9,
  'optimizer'         : 'SGD',
# 'scheduler'         : 'mannual',
  'scheduler'         : FactorScheduler(initial_lr, 0.1, int(1E5)),
  'weight_decay'      : 0.0001
}

print 'loading data'
# data = load_whitened_cifar10(path='../utilities/whitened-cifar', reshape=True)
data = load_cifar10(path='../utilities/cifar/', reshape=True, center=True, rescale=True)
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

activation = 'ReLU'
ini_mode = 'layer-by-layer'
# ini_mode = 'normal'
ini_file = 'NIN-%s-%s-ini' % (activation, ini_mode)
initial_parameters = pickle.load(open(ini_file, 'rb'))
model = NIN(activation, DReLUInitializer(dictionary=initial_parameters))
solver = MXSolver(model, **solver_configuration)
history = solver.train() # test_accuracy, progress
path = 'NIN-%s' % activation
pickle.dump(history, open('../models/%s' % path, 'wb'))
