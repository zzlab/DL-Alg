import sys
sys.path.append('../../nn')
from GPU_utility import *

from utilities.data_utility import *

import MXModels
from MXInitializer import *
import MXSolver

import argparse
import cPickle as pickle

data = load_cifar10(reshape=True, center=True, rescale=True)

parser = argparse.ArgumentParser()

parser.add_argument('--activation')
parser.add_argument('--convolution-layers', type=int)

args = parser.parse_args()

activation = args.activation
convolution_layers = args.convolution_layers

settings = (((2, 2), 64, (2, 2)),) * 2 + (((3, 3), 32, (1, 1), (1, 1)),) * convolution_layers

model = MXModels.CNN(settings, activation, DReLUInitializer())

optimizer_settings = {
  'lr'                : 0.05,
  'lr_decay_interval' : 1,
  'lr_decay_factor'   : 1.0,
  'optimizer'         : 'SGD',
  'weight_decay'      : 0
}

solver_configuration = {
  'batch_size'         : 128,
  'data'               : data,
  'devices'            : GPU_availability()[:4],
  'epoch'              : 100,
  'optimizer_settings' : optimizer_settings,
}

path = 'CIFAR-10-%d-layer-no-pooling-CNN-%s' % (convolution_layers + 2, activation)

solver = MXSolver.MXSolver(model, **solver_configuration)
history = solver.train()
pickle.dump(history, open('models/%s-history' % path, 'wb'))
pickle.dump((solver.model.arg_params, solver.model.aux_params), open('models/%s-parameters' % path, 'wb'))
