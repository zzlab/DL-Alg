import sys

sys.path.append('../')
sys.path.append('../../nn')

from utilities.GPU_utility import GPU_availability
from utilities.data_utility import load_cifar10
from MXModels.ResidualNetwork import ResidualNetwork, ResidualNetworkScheduler
from MXInitializer import DReLUInitializer
from MXSolver import MXSolver

import cPickle as pickle

args = dict(enumerate(sys.argv))
activation = args.pop(1, 'BNReLU')
n = int(args.pop(2, '3'))
kernel = args.pop(3, 'double')
path = args.pop(4, None)

data = load_cifar10(path='../utilities/cifar/', reshape=True, center=True, rescale=True)

optimizer_settings = {
  'lr'                : 0.1,
  'momentum'          : 0.9,
  'optimizer'         : 'SGD',
  'scheduler'         : 'mannual',
# 'scheduler'         : ResidualNetworkScheduler(),
  'weight_decay'      : 0.0001
}

solver_configuration = {
  'batch_size'         : 128,
  'data'               : data,
  'devices'            : GPU_availability()[:4],
  'epoch'              : 50,
  'file'               : '../../nn/lr',
  'optimizer_settings' : optimizer_settings,
  'verbose'            : True
}

model = ResidualNetwork(n, activation, DReLUInitializer(), kernel=='double')
solver = MXSolver(model, **solver_configuration)
history = solver.train() # test_accuracy, progress
if path:
  pickle.dump(history, open('history/%s-history' % path, 'wb'))
  pickle.dump(
    (model, solver.model.arg_params, solver.model.aux_params),
    open('models/%s-parameter' % path, 'wb')
  )
