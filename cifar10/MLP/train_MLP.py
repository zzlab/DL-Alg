import sys
sys.path.append('../')
sys.path.append('../../nn')
from MXInitializer import DReLUInitializer
from MXModels.MLP import MLP
from MXSolver import MXSolver
from utilities.data_utility import load_cifar10
from utilities.GPU_utility import GPU_availability

import cPickle as pickle

args = dict(enumerate(sys.argv))
activation = args.pop(1, 'DReLU')
hidden_layers = int(args.pop(2, '1'))
device = [int(args.pop(3, GPU_availability()[0]))]

shape = (1024,) * hidden_layers + (10,)

data = load_cifar10(path='../utilities/cifar/', center=True, rescale=True)

model = MLP(shape, activation, data[0])
# model = MLP(shape, activation)

optimizer_settings = {
  'lr'                : 0.5,
  'optimizer'         : 'SGD',
  'scheduler'          : 'mannual',
}

solver_configuration = {
  'batch_size'         : 128,
  'data'               : data,
  'devices'            : device,
  'epoch'              : 200,
  'file'               : '../../nn/lr',
  'optimizer_settings' : optimizer_settings,
  'verbose'            : True,
}

path = ('MLP-%s10-%s' % ('{}-' * (len(shape) - 1), activation)).format(*shape[:-1])

solver = MXSolver(model, **solver_configuration)
test_accuracy, progress = solver.train()

pickle.dump((test_accuracy, progress), open('models/%s-history' % path, 'wb'))
pickle.dump((solver.model.arg_params, solver.model.aux_params), open('models/%s-parameters' % path, 'wb'))
