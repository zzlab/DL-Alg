import minpy.numpy as np
import minpy.nn.model_builder as builder

from minpy.context import set_context, cpu, gpu
# set_context(gpu(0))
set_context(cpu())

import cPickle as pickle

import sys
sys.path.append('../../nn')
from custom_layers import *
from facility import *
from solver_primitives import *

from loss import stochastic_gradient_loss

sys.path.append('../')
from utilities.data_utility import load_mnist
X, Y, _, _, _, _, = load_mnist(path='../utilities')

hidden_layers = 4
shapes = (1024,) * hidden_layers + (10,)
activation = builder.ReLU
storage = {}
mlp = builder.Sequential()
for i, shape in enumerate(shapes[:-1]):
  mlp.append(builder.Affine(shape))
  mlp.append(builder.Export('affine%d' % i, storage))
  mlp.append(activation())
mlp.append(builder.Affine(shapes[-1]))
mlp.append(builder.Export('affine%d' % (len(shapes) - 1), storage))
model = builder.Model(mlp, 'softmax', (28 * 28,))

initialize(model)
rescale(mlp, X[:1024], model.params)

p = {key : value.asnumpy() for key, value in model.params.items()}
for key, value in p.items():
  if 'weight' in key:
    p[key] = value.T

pickle.dump(p, open('mlp-initial-parameters', 'wb'))
