import sys
sys.path.append('../')
sys.path.append('../../nn')
from MXInitializer import DReLUInitializer
from MXModels.MLP import MLP
from MXSolver import MXSolver
from utilities.data_utility import load_mnist
from utilities.GPU_utility import GPU_availability

import cPickle as pickle

HIDDEN_LAYERS = int(sys.argv[1])
ACTIVATION = sys.argv[2]
DEVICE = [int(sys.argv[3])]

data = load_mnist(path='../utilities')
initial_parameters = pickle.load(open('mlp-initial-parameters', 'rb'))
initial_parameters['data'], initial_parameters['softmax_label'] = data[:2]
for key, value in initial_parameters.items():
  print key, value.shape
shape = (1024,) * HIDDEN_LAYERS + (10,)
model = MLP(shape, ACTIVATION, DReLUInitializer(dictionary=initial_parameters))

optimizer_settings = {
  'lr'                : 0.1,
  'optimizer'         : 'SGD',
  'lr_decay_factor'   : 1,
  'lr_decay_interval' : 1,
}

solver_configuration = {
  'batch_size'         : 200,
  'data'               : data,
  'devices'            : DEVICE,
  'epoch'              : 10,
  'file'               : '../../nn/lr',
  'optimizer_settings' : optimizer_settings,
  'verbose'            : True,
}

solver = MXSolver(model, **solver_configuration)
test_accuracy, progress = solver.train()

'''
path = ('MLP-%s10-%s' % ('{}-' * (len(shape) - 1), ACTIVATION)).format(*shape[:-1])
pickle.dump((test_accuracy, progress), open('models/%s-history' % path, 'wb'))
pickle.dump((solver.model.arg_params, solver.model.aux_params), open('models/%s-parameters' % path, 'wb'))
'''
