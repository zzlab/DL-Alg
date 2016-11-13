import numpy as np

import sys
sys.path.append('../')
sys.path.append('../../nn')
from MXInitializer import DReLUInitializer
from MXModels.MLP import MLP
from MXSolver import MXSolver
from utilities.data_utility import load_cifar10
from utilities.GPU_utility import GPU_availability

import cPickle as pickle

ACTIVATION = sys.argv[1]
DEVICE = [int(sys.argv[2])]
SHAPES = [int(shape) for shape in sys.argv[3:]]

data = load_cifar10(path='../utilities/cifar/', center=True, rescale=True)

configuration = '-'.join(sys.argv[1 : 2] + sys.argv[3:])
p = pickle.load(open('model/MLP-initial-%s' % configuration, 'rb'))
p['data'] = data[0]
p['softmax_label'] = data[1]

model = MLP(SHAPES, ACTIVATION, DReLUInitializer(dictionary=p))

optimizer_settings = {
  'lr'        : 0.05,
  'optimizer' : 'SGD',
  'scheduler' : 'mannual',
}

solver_configuration = {
  'batch_size'         : 100,
  'data'               : data,
  'devices'            : DEVICE,
  'epoch'              : 50,
  'file'               : '../../nn/lr',
  'optimizer_settings' : optimizer_settings,
  'verbose'            : True,
}

solver = MXSolver(model, **solver_configuration)
test_accuracy, progress = solver.train()

pickle.dump((test_accuracy, progress), open('model/MLP-%s-history' % configuration, 'wb'))
'''
pickle.dump(
  (solver.model.arg_params, solver.model.aux_params),
  open('models/MLP-%s-parameters' % configuration, 'wb')
)
'''
