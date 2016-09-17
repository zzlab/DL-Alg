import sys
sys.path.append('../')
sys.path.append('../../nn')

from utilities.data_utilities import load_cifar10
from utilities.GPU_utility import GPU_availability

import MXModels
from MXInitializer import *
import MXSolver

import argparse
import cPickle as pickle

data = load_cifar10(reshape=True, center=True, rescale=True)

parser = argparse.ArgumentParser()

parser.add_argument('--activation')
parser.add_argument('--n', type=int)

args = parser.parse_args()

activation = 'BNReLU'
n = 3

optimizer_settings = {
  'lr'                : 0.001,
  'lr_decay_interval' : 20,
  'lr_decay_factor'   : 0.9,
  'optimizer'         : 'Adam',
  'weight_decay'      : 0
}

solver_configuration = {
  'batch_size'         : 128,
  'data'               : data,
  'devices'            : GPU_availability()[:4],
  'epoch'              : 100,
  'optimizer_settings' : optimizer_settings,
}

kernel = 'double'
model = MXModels.ResidualNetwork(n, activation, DReLUInitializer(), kernel=='double')
solver = MXSolver.MXSolver(model, **solver_configuration)
history = solver.train() # test_accuracy, progress
path = 'residual-%d-%s-%s-Adam' % (n, activation, kernel)
pickle.dump(history, open('models/%s-history' % path, 'wb'))
# pickle.dump((solver.model.arg_params, solver.model.aux_params), open('models/%s-parameters' % path, 'wb'))
