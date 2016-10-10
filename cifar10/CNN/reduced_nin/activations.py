import sys
sys.path.append('../..')
sys.path.append('../../nn')

import cPickle as pickle
import mxnet as mx
import numpy as np

from utilities.GPU_utility import GPU_availability
from utilities.data_utility import load_cifar10

from symbol import ReducedNIN

device = mx.gpu(GPU_availability()[0])

data = load_cifar10(path='../../utilities/cifar/', reshape=True, center=True, rescale=True)

prefix = 'reduced-nin'
args, auxes = pickle.load(open('models/%s-parameters' % prefix, 'rb'))

for parameters in args, auxes:
  for key, value in parameters.items():
    parameters[key] = mx.nd.zeros(value.shape, device)
    print key, 'allocated'
    value.copyto(parameters[key])

# only test data
args['data'] = mx.nd.array(data[2], device)

outputs = {}

symbol = ReducedNIN(
  'DReLU',
  filter_count,
  Initializer.DReLUInitializer(3.0)
)(
  data[2].shape,
  True
)
executor = symbol.bind(device, args=args, aux_states=auxes)

print 'training data'
executor.forward(is_train=False)
outputs = tuple(output.asnumpy() for output in executor.outputs)

pickle.dump(outputs, open('activations/%s-post' % prefix, 'wb'))
