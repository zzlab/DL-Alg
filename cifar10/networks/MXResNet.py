import sys
<<<<<<< HEAD

sys.path.append('../')
sys.path.append('../../nn')
=======
sys.path.append('../')
sys.path.append('../../nn')

from utilities.data_utilities import load_cifar10
from utilities.GPU_utility import GPU_availability
>>>>>>> 15eae9102bfff20c72efb5a0be4fe41fca98378f

from utilities.GPU_utility import GPU_availability
from utilities.data_utility import load_cifar10
from MXModels.ResidualNetwork import ResidualNetwork
from MXInitializer import DReLUInitializer
from MXSolver import MXSolver

import cPickle as pickle

args = dict(enumerate(sys.argv))
activation = args.pop(1, 'BNReLU')
n = int(args.pop(2, '3'))

<<<<<<< HEAD
data = load_cifar10(path='../utilities/cifar/', reshape=True, center=True, rescale=True)

optimizer_settings = {
  'lr'                : 0.001,
  'lr_decay_interval' : 1,
  'lr_decay_factor'   : 1.0,
# 'momentum'          : 0.3,
=======
optimizer_settings = {
  'lr'                : 0.001,
  'lr_decay_interval' : 20,
  'lr_decay_factor'   : 0.9,
>>>>>>> 15eae9102bfff20c72efb5a0be4fe41fca98378f
  'optimizer'         : 'Adam',
  'weight_decay'      : 0
}

solver_configuration = {
  'batch_size'         : 128,
  'data'               : data,
  'devices'            : GPU_availability()[:4],
  'epoch'              : 300,
  'file'               : '../../nn/lr',
  'optimizer_settings' : optimizer_settings,
  'verbose'            : True
}

<<<<<<< HEAD
path = 'residual-%d-%s' % (n, activation)

model = ResidualNetwork(n, activation, DReLUInitializer())
solver = MXSolver(model, **solver_configuration)
test_accuracy, progress = solver.train()
pickle.dump(progress, open('history/%s-history' % path, 'wb'))
=======
kernel = 'double'
model = MXModels.ResidualNetwork(n, activation, DReLUInitializer(), kernel=='double')
solver = MXSolver.MXSolver(model, **solver_configuration)
history = solver.train() # test_accuracy, progress
path = 'residual-%d-%s-%s-Adam' % (n, activation, kernel)
pickle.dump(history, open('models/%s-history' % path, 'wb'))
# pickle.dump((solver.model.arg_params, solver.model.aux_params), open('models/%s-parameters' % path, 'wb'))
>>>>>>> 15eae9102bfff20c72efb5a0be4fe41fca98378f
