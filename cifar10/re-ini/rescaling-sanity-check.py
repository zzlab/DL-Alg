import minpy.numpy as np
import minpy.nn.model_builder as builder

from minpy.context import set_context, cpu, gpu
# set_context(gpu(0))
set_context(cpu())

import sys
sys.path.append('../../nn')
from custom_layers import *
from facility import *
from solver_primitives import *

sys.path.append('../')
from utilities.data_utility import load_cifar10
data = load_cifar10(path='../utilities/cifar/', center=True, rescale=True)

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

model = builder.Model(mlp, 'softmax', (3072,))

for key, value in model.param_configs.items():
  if 'weight' in key:
    value['init_rule'] = 'gaussian'
    value['init_config'] = {'stdvar' : 1}

initialize(model)
for key, value in model.params.items():
  if 'weight' in key:
    print np.std(value)

rescale(mlp, data[2], model.params) # validation data
for key, value in model.params.items():
  if 'weight' in key:
    print np.std(value)

result = model.forward(data[2], 'train')
print np.std(result)
