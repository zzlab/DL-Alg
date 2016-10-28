import sys
sys.path.append('../')
sys.path.append('../../nn')

import cPickle as pickle
import marshal

from utilities.data_utility import load_cifar10
from utilities.data_utility import load_whitened_cifar10
from utilities.GPU_utility import GPU_availability
from LRScheduler import DecayingAtEpochScheduler
from MXInitializer import DReLUInitializer
from MXModels.ResidualNetwork import ResidualNetwork
from MXSolver import *
from MXInitializer import *

print 'loading data'
# data = load_whitened_cifar10(path='../utilities/whitened-cifar', reshape=True)
data = load_cifar10(path='../utilities/cifar/', reshape=True, center=True, rescale=True)
print 'data loaded'

epoch_to_decay = [int(sys.argv[1])]
batch_size = 128
batches = len(data[0]) // batch_size

initial_lr = 0.01
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
  'epoch'              : 200,
  'file'               : '../../nn/lr',
  'optimizer_settings' : optimizer_settings,
  'verbose'            : False
}

# activation = 'DReLU'
activation = sys.argv[2]
ini_file = 'resnet-%s-initial-parameters' % (activation)
initial_parameters = pickle.load(open(ini_file, 'rb'))
model = ResidualNetwork(
  3,
  activation,
  DReLUInitializer(dictionary=initial_parameters)
)
solver = MXSolver(model, **solver_configuration)
history = solver.train() # test_accuracy, progress
path = 'resnet_lr_decay_epoch_0_grid_search/%s-decay-at-epoch-%d' % (activation, epoch_to_decay[0])
pickle.dump(history, open(path, 'wb'))
