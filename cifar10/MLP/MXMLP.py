import sys
sys.path.append('../../nn')
from GPU_utility import *

from data_utilities import *

import MXModels
from MXInitializer import *
import MXSolver

import argparse
import cPickle as pickle

data = load_cifar10(center=True, rescale=True)

parser = argparse.ArgumentParser()

parser.add_argument('--activation')
parser.add_argument('--hidden-layers', type=int)

args = parser.parse_args()

activation = args.activation
hidden_layers = args.hidden_layers

shape = (1024,) * hidden_layers + (10,)
model = MXModels.MLP(shape, activation, MXModels.DReLUInitializer())

optimizer_settings = {
  'lr'                : 0.1,
  'lr_decay_interval' : 1,
  'lr_decay_factor'   : 1.0,
  'optimizer'         : 'SGD',
  'weight_decay'      : 0
}

solver_configuration = {
  'batch_size'         : 128,
  'data'               : data,
  'devices'            : GPU_availability()[:1],
  'epoch'              : 50,
  'optimizer_settings' : optimizer_settings,
}

path = ('CIFAR-%s10-%s' % ('{}-' * (len(shape) - 1), activation)).format(*shape[:-1])

solver = MXSolver.MXSolver(model, **solver_configuration)
history, test = solver.train()

pickle.dump(history, open('models/%s-history' % path, 'wb'))
pickle.dump((solver.model.arg_params, solver.model.aux_params), open('models/%s-parameters' % path, 'wb'))
