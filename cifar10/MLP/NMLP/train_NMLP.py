import sys
sys.path.append('../../')
sys.path.append('../../../nn')
from MXInitializer import DReLUInitializer
from MXSolver import MXSolver
from utilities.data_utility import load_cifar10
from utilities.GPU_utility import GPU_availability

import cPickle as pickle

args = dict(enumerate(sys.argv))
model_label = args.pop(1, 'CMLP')
hidden_layers = int(args.pop(2, '1'))
device = [int(args.pop(3, GPU_availability()[0]))]

shape = (1024,) * hidden_layers + (10,)
import MXModels.NMLP as NMLP
model = getattr(NMLP, model_label)(shape, DReLUInitializer())

optimizer_settings = {
  'lr'                : 0.05,
  'lr_decay_interval' : 1,
  'lr_decay_factor'   : 1,
  'optimizer'         : 'SGD',
  'weight_decay'      : 0
}

data = load_cifar10(path='../../utilities/cifar/', center=True, rescale=True)

def callback(epoch, symbol, args, auxes):
  for key, value in args.items():
    value = value.asnumpy()
    # print key, value.min(), value.max()
  
solver_configuration = {
  'batch_size'         : 128,
  'callback'           : callback,
  'data'               : data,
  'devices'            : device,
  'epoch'              : 50,
  'file'               : '../../../nn/lr',
  'optimizer_settings' : optimizer_settings,
  'verbose'            : False
}

path = ('%s-%s10' % (model_label, '{}-' * (len(shape) - 1))).format(*shape[:-1])

solver = MXSolver(model, **solver_configuration)
test_accuracy, progress = solver.train()

pickle.dump((test_accuracy, progress), open('models/%s-history' % path, 'wb'))
pickle.dump((solver.model.arg_params, solver.model.aux_params), open('models/%s-parameters' % path, 'wb'))
