import sys
sys.path.append('../../nn')
from GPU_utility import *

from data_utilities import *

import MXModels
from MXInitializer import *
import MXSolver

import argparse
import cPickle as pickle

data = load_cifar10(reshape=True, center=True, rescale=True)

activation = sys.argv[1]
convolution_layers = int(sys.argv[2])
optimizer = sys.argv[3]
print optimizer
if len(sys.argv) > 4:
  other_settings = {sys.argv[4] : sys.argv[5]}

settings = (((2, 2), 64, (2, 2)),) * 2 + (((3, 3), 32, (1, 1), (1, 1)),) * convolution_layers

model = MXModels.CNN(settings, activation, DReLUInitializer())

optimizer_settings = {
  'lr'                : 0.001,
  'lr_decay_interval' : 1,
  'lr_decay_factor'   : 1.0,
  'optimizer'         : optimizer,
  'weight_decay'      : 0
}

if len(sys.argv) > 4:
  optimizer_settings.update(other_settings)

solver_configuration = {
  'batch_size'         : 128,
  'data'               : data,
  'devices'            : GPU_availability()[:4],
  'epoch'              : 50,
  'optimizer_settings' : optimizer_settings,
}

path = 'CIFAR-10-%d-layer-no-pooling-CNN-%s-%s' % (convolution_layers + 2, activation, str(optimizer_settings))

solver = MXSolver.MXSolver(model, **solver_configuration)
history = solver.train()
pickle.dump(history, open('models/%s-history' % path, 'wb'))
pickle.dump((solver.model.arg_params, solver.model.aux_params), open('models/%s-parameters' % path, 'wb'))
