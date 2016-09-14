import sys
sys.path.append('../../')
sys.path.append('../../../nn')
import mxnet as mx
from MXInitializer import DReLUInitializer
from MXModels.MLP import MLP as MLP
from MXSolver import MXSolver
from utilities.data_utility import load_cifar10
from utilities.GPU_utility import GPU_availability

import cPickle as pickle

args = dict(enumerate(sys.argv))
activation = args.pop(1, 'ReLU')
hidden_layers = int(args.pop(2, '1'))
device = int(args.pop(3, GPU_availability()[0]))
device = mx.gpu(device)

shape = (1024,) * hidden_layers + (10,)

model = MLP(shape, activation, DReLUInitializer())

data = load_cifar10(path='../../utilities/cifar/', center=True, rescale=True)

symbol, _ = model(data[0].shape, True)

path = ('MLP-%s10-DReLU' % ('{}-' * (len(shape) - 1))).format(*shape[:-1])
args, _ = pickle.load(open('../models/%s-parameters' % path, 'rb'))
args.update({'data' : mx.nd.array(data[0], device)})

record = {}

executor = symbol.bind(device, args)
executor.forward()
record.update({'training' : executor.outputs[:-1]})
print 'training data finished'

executor = executor.reshape(data=data[2].shape)
executor.forward(data=mx.nd.array(data[2], device))
record.update({'validation' : executor.outputs[:-1]})
print 'validation data finished'

executor = executor.reshape(data=data[4].shape)
executor.forward(data=mx.nd.array(data[4], device))
record.update({'test' : executor.outputs[:-1]})
print 'test data finished'

pickle.dump(record, open('models/%s-activation' % path, 'wb'))
