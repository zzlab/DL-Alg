import cPickle as pickle
import sys

from lr_scheduler import AtEpochScheduler, AtIterationScheduler
from data_utilities import load_cifar10_record
from mx_layers import ReLU
from mx_solver import MXSolver
from drelu import drelu, DReLUInitializer

from residual_network import residual_network

N = int(sys.argv[1])
ACTIVATE = sys.argv[2]
BATCH_SIZE = 128
if ACTIVATE == 'relu' : activate = ReLU
elif ACTIVATE == 'drelu': activate = lambda X : drelu(X, {'data' : (BATCH_SIZE, 3, 32, 32)})
TIMES = int(sys.argv[3])
network = residual_network(N, activate, TIMES)

lr = 0.1
lr_table = {32000 : lr * 0.1, 48000 : lr * 0.01}
lr_scheduler = AtIterationScheduler(lr, lr_table)
optimizer_settings = {
  'args'         : {'momentum' : 0.9},
  'initial_lr'   : lr,
  'lr_scheduler' : lr_scheduler,
  'optimizer'    : 'SGD',
  'weight_decay' : 0.0001,
}

solver = MXSolver(
  batch_size = BATCH_SIZE,
  devices = (0, 1, 2, 3),
  epochs = 150,
  initializer = DReLUInitializer(0, 1),
  optimizer_settings = optimizer_settings,
  symbol = network,
  verbose = True,
)

data = load_cifar10_record(BATCH_SIZE)
info = solver.train(data)

identifier = 'residual-network-n-%d-activate-%s-times-%d' % (N, ACTIVATE, TIMES)
pickle.dump(info, open('info/%s' % identifier, 'wb'))
parameters = solver.export_parameters()
pickle.dump(parameters, open('parameters/%s' % identifier, 'wb'))
