import sys
sys.path.append('../../')
sys.path.append('../../../nn')
import mxnet as mx
from MXInitializer import DReLUInitializer
import MXModels.NMLP as NMLP
from MXSolver import MXSolver
from utilities.data_utility import load_cifar10
from utilities.GPU_utility import GPU_availability

import cPickle as pickle

args = dict(enumerate(sys.argv))
model_label = args.pop(1, 'CMLP')
hidden_layers = int(args.pop(2, '3'))
device = int(args.pop(3, GPU_availability()[0]))
device = mx.gpu(device)

shape = (1024,) * hidden_layers + (10,)

model = getattr(NMLP, model_label)(shape, DReLUInitializer())

data = load_cifar10(path='../../utilities/cifar/', center=True, rescale=True)

symbol = model(data[0].shape, True)

path = ('%s-%s10' % (model_label, '{}-' * (len(shape) - 1))).format(*shape[:-1])
args, _ = pickle.load(open('models/%s-parameters' % path, 'rb'))
args.update({'data' : mx.nd.array(data[0], device)})

record = {
  'training'   : [],
  'validation' : [],
  'test'       : []
}

executor = symbol.bind(device, args)
executor.forward()
for index, output in enumerate(executor.outputs[:-2]):
  if index % 2 == 1:
    lower = args['NDReLU_lower%d' % (index // 2)]
    upper = args['NDReLU_upper%d' % (index // 2)]
    weight = args['fullyconnected%d_weight' % (index // 2 + 1)]
    inferred_mean = mx.nd.dot((lower + upper) / 2, weight)
    mean = mx.nd.sum(output, axis=0) / float(len(data[0]))
    error = mx.nd.dot(inferred_mean - mean, weight)
    record['training'].append(error.asnumpy()) 
print 'training data finished'

executor = executor.reshape(data=data[2].shape)

executor.forward(data=mx.nd.array(data[2], device))
for index, output in enumerate(executor.outputs[:-2]):
  if index % 2 == 1:
    lower = args['NDReLU_lower%d' % (index // 2)]
    upper = args['NDReLU_upper%d' % (index // 2)]
    weight = args['fullyconnected%d_weight' % (index // 2 + 1)]
    inferred_mean = mx.nd.dot((lower + upper) / 2, weight)
    mean = mx.nd.sum(output, axis=0) / float(len(data[2]))
    error = mx.nd.dot(inferred_mean - mean, weight)
    record['validation'].append(error.asnumpy())
print 'validation data finished'

executor = executor.reshape(data=data[4].shape)
executor.forward(data=mx.nd.array(data[4], device))
for index, output in enumerate(executor.outputs[:-2]):
  if index % 2 == 1:
    lower = args['NDReLU_lower%d' % (index // 2)]
    upper = args['NDReLU_upper%d' % (index // 2)]
    weight = args['fullyconnected%d_weight' % (index // 2 + 1)]
    inferred_mean = mx.nd.dot((lower + upper) / 2, weight)
    mean = mx.nd.sum(output, axis=0) / float(len(data[4]))
    error = mx.nd.dot(inferred_mean - mean, weight)
    record['test'].append(error.asnumpy()) 
print 'test data finished'

pickle.dump(record, open('models/%s-error-of-mean-affine' % path, 'wb'))
