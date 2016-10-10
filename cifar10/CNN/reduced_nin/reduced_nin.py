import sys
sys.path.append('../..')
sys.path.append('../../../nn')

import utilities.data_utility as data_utility
import utilities.GPU_utility as GPU_utility

import MXModels as Models
import MXInitializer as Initializer
import MXSolver as Solver

from callbacks.checkpoint import BoundsCheckpoint

from symbol import ReducedNIN

import cPickle as pickle

'''
activation = sys.argv[1]
filter_count = int(sys.argv[2])
label = sys.argv[3]
'''

activation = 'DReLU'
filter_count = 64
label = 'reduced-nin'

model = ReducedNIN(activation, filter_count, Initializer.DReLUInitializer(2.0))

optimizer_settings = {
  'lr'                : 0.05,
  'optimizer'         : 'SGD',
  'scheduler'         : 'mannual',
  'weight_decay'      : 0.0
}

data = data_utility.load_cifar10(path='../../utilities/cifar/', reshape=True, center=True, rescale=True)

callback = BoundsCheckpoint(label, 'models')

solver_configuration = {
  'batch_size'         : 128,
# 'batch_end_callback' : callback.batch_end_callback,
  'data'               : data,
  'devices'            : GPU_utility.GPU_availability()[:3],
  'epoch'              : 1,
# 'epoch_end_callback' : callback.epoch_end_callback,
  'file'               : '../../../nn/lr',
  'optimizer_settings' : optimizer_settings,
  'verbose'            : True,
}

solver = Solver.MXSolver(model, **solver_configuration)
history = solver.train()
pickle.dump(history, open('history/%s' % label, 'wb'))
pickle.dump((solver.model.arg_params, solver.model.aux_params), open('models/%s-parameters' % label, 'wb'))
