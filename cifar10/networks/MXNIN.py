import sys
sys.path.append('../../nn')

import cPickle as pickle

from data_utilities import *
from GPU_utility import *
from MXModels import *
from MXSolver import *
from MXInitializer import *

import sys
m = float(sys.argv[1])
print m

data = load_cifar10(reshape=True, center=True, rescale=True)

model = NIN('DReLU', DReLUInitializer())

optimizer_settings = {
  'lr'                : 0.2,
  'lr_decay_interval' : 10,
  'lr_decay_factor'   : 1.0,
  'momentum'          : m,
  'optimizer'         : 'SGD',
  'weight_decay'      : 0
}

solver_configuration = {
  'batch_size'         : 128,
  'data'               : data,
  'devices'            : GPU_availability()[:4],
  'epoch'              : 200,
  'optimizer_settings' : optimizer_settings,
}

solver = MXSolver(model, **solver_configuration)
history = solver.train()
path = 'NIN-{}-history'.format(optimizer_settings)
pickle.dump(history, open('models/%s' % path, 'wb'))
