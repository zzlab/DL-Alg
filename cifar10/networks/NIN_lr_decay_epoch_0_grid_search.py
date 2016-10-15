import sys
sys.path.append('../')
sys.path.append('../../nn')

import cPickle as pickle

from utilities.data_utility import load_cifar10
from utilities.data_utility import load_whitened_cifar10
from utilities.GPU_utility import GPU_availability
from LRScheduler import DecayingAtEpochScheduler
from MXModels.NIN import NIN
from MXSolver import *
from MXInitializer import *

print 'loading data'
# data = load_whitened_cifar10(path='../utilities/whitened-cifar', reshape=True)
data = load_cifar10(path='../utilities/cifar/', reshape=True, center=True, rescale=True)
print 'data loaded'

epoch_to_decay = [int(sys.argv[1])]
batch_size = 128
batches = len(data[0]) // batch_size

initial_lr = 0.1
scheduler = DecayingAtEpochScheduler(initial_lr, 0.1, epoch_to_decay, batches)
optimizer_settings = {
  'lr'                : initial_lr,
  'momentum'          : 0.9,
  'optimizer'         : 'SGD',
  'scheduler'         : scheduler,
  'weight_decay'      : 0.0001
}

solver_configuration = {
  'batch_size'         : batch_size,
  'data'               : data,
  'devices'            : GPU_availability()[:4],
  'epoch'              : 5,
  'file'               : '../../nn/lr',
  'optimizer_settings' : optimizer_settings,
  'verbose'            : True
}

activation = 'ReLU'
# ini_mode = 'layer-by-layer'
ini_mode = 'normal'
ini_file = 'NIN-%s-%s-initial-parameters' % (activation, ini_mode)
initial_parameters = pickle.load(open(ini_file, 'rb'))
model = NIN(activation, DReLUInitializer(dictionary=initial_parameters))
solver = MXSolver(model, **solver_configuration)
history = solver.train() # test_accuracy, progress
path = 'NIN_lr_decay_epoch_0_grid_search/%s-decay-at-epoch-%d' % (activation, epoch_to_decay[0])
pickle.dump(history, open(path, 'wb'))
