import sys
sys.path.append('../')
sys.path.append('../../nn')

import cPickle as pickle
import mxnet as mx
import numpy as np

from MXModels.ResidualNetwork import ResidualNetwork, ResidualNetworkScheduler

from utilities.GPU_utility import GPU_availability
from utilities.data_utility import load_cifar10

device = mx.gpu(GPU_availability()[0])

data = load_cifar10(path='../utilities/cifar/', reshape=True, center=True, rescale=True)

prefix = sys.argv[1]
model, args, auxes = pickle.load(open('models/%s-parameter' % prefix, 'rb'))

for parameters in args, auxes:
  for key, value in parameters.items():
    print key
    parameters[key] = mx.nd.zeros(value.shape, device)
    print 'allocated'
    value.copyto(parameters[key])

args['data'] = mx.nd.array(data[2], device)

outputs = {}

# symbol = ResidualNetwork(3, 'BNDReLU', None, True)(data[2].shape, True)
symbol = model(data[0].shape, 'post-activation')
executor = symbol.bind(device, args=args, aux_states=auxes)

print 'training data'
executor.forward(is_train=False)
# outputs['training'] = tuple(output.asnumpy() for output in executor.outputs)

executor = executor.reshape(data=data[2].shape)
args['data'] = mx.nd.array(data[2], device)

print 'validation data'
executor.forward(is_train=False)
# outputs['validation'] = tuple(output.asnumpy() for output in executor.outputs)

args['data'] = mx.nd.array(data[4], device)
print 'test data'
executor.forward(is_train=False)
outputs['test'] = tuple(output.asnumpy().astype(np.float32) for output in executor.outputs)
print 'finished'

pickle.dump(outputs, open('models/%s-post-activations' % prefix, 'wb'))
