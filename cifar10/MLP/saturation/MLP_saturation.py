import sys
sys.path.append('../../nn')
from GPU_utility import *

from data_utilities import *

import mxnet as mx
import MXModels
import MXSolver

import argparse
import cPickle as pickle
import numpy as np

data = load_cifar10(center=True, rescale=True)

parser = argparse.ArgumentParser()

parser.add_argument('--activation')
parser.add_argument('--hidden-layers', type=int)

args = parser.parse_args()

activation = args.activation
hidden_layers = args.hidden_layers
shape = (1024,) * hidden_layers + (10,)
model = MXModels.MLP(shape, activation, MXModels.DReLUInitializer())

path = ('CIFAR-%s10-%s' % ('{}-' * (len(shape) - 1), activation)).format(*shape[:-1])

symbol, _ = model(data[0].shape, True)
args, auxes = pickle.load(open('models/%s-parameters' % path, 'rb'))
args.update({'data' : mx.nd.array(data[0])})
executor = symbol.bind(mx.gpu(GPU_availability()[0]), args=args, aux_states=auxes)
executor.forward()
outputs = executor.outputs
# print len(outputs), outputs
layer_width = tuple(float(d) for d in shape[:-1])
training_record = []
for layer in range(hidden_layers):
  activated_output = outputs[layer].asnumpy()
  lower = activated_output.min(axis=0)
  upper = activated_output.max(axis=0)
  if 'DReLU' in activation:
    '''
    lower = args['lower'].asnumpy()
    upper = args['upper'].asnumpy()
    '''
    lower_saturation_rate = np.sum(activated_output == lower) / (layer_width[layer] * data[0].shape[0])
    upper_saturation_rate = np.sum(activated_output == upper) / (layer_width[layer] * data[0].shape[0])
    training_record.append((lower_saturation_rate, upper_saturation_rate))
  else:
    saturation_rate = np.sum(activated_output == lower) / (layer_width[layer] * data[0].shape[0])
    training_record.append(saturation_rate)

executor = executor.reshape(data=data[2].shape)
executor.forward(is_train=False, data=mx.nd.array(data[2]))
outputs = executor.outputs
validation_record = []
for layer in range(hidden_layers):
  activated_output = outputs[layer].asnumpy()
  lower = activated_output.min(axis=0)
  upper = activated_output.max(axis=0)
  if 'DReLU' in activation:
    lower_saturation_rate = np.sum(activated_output == lower) / (layer_width[layer] * data[2].shape[0])
    upper_saturation_rate = np.sum(activated_output == upper) / (layer_width[layer] * data[2].shape[0])
    validation_record.append((lower_saturation_rate, upper_saturation_rate))
  else:
    saturation_rate = np.sum(activated_output == lower) / (layer_width[layer] * data[2].shape[0])
    validation_record.append(saturation_rate)

executor = executor.reshape(data=data[4].shape)
executor.forward(is_train=False, data=mx.nd.array(data[4]))
outputs = executor.outputs
test_record = []
for layer in range(hidden_layers):
  activated_output = outputs[layer].asnumpy()
  lower = activated_output.min(axis=0)
  upper = activated_output.max(axis=0)
  if 'DReLU' in activation:
    lower_saturation_rate = np.sum(activated_output == lower) / (layer_width[layer] * data[4].shape[0])
    upper_saturation_rate = np.sum(activated_output == upper) / (layer_width[layer] * data[4].shape[0])
    test_record.append((lower_saturation_rate, upper_saturation_rate))
  else:
    saturation_rate = np.sum(activated_output == lower) / (layer_width[layer] * data[4].shape[0])
    test_record.append(saturation_rate)

pickle.dump((training_record, validation_record, test_record), open('models/%s-saturation' % path, 'wb'))
