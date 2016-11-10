import minpy.numpy as np
import minpy.nn.model_builder as builder

from minpy.context import set_context, cpu
set_context(cpu())

import sys
sys.path.append('../../nn')
from facility import *
from solver_primitives import *

HIDDEN_LAYERS = 4
shapes = (1024,) * HIDDEN_LAYERS + (10,)
storage = {}
activation = builder.ReLU
mlp = builder.Sequential()
for i, shape in enumerate(shapes[:-1]):
  mlp.append(builder.Affine(shape))
  mlp.append(builder.Export('affine%d' % i, storage))
  mlp.append(activation())
mlp.append(builder.Affine(shapes[-1]))
mlp.append(builder.Export('affine%d' % (len(shapes) - 1), storage))
model = builder.Model(mlp, 'softmax', (3072,))

initialize(model)

X = np.random.normal(0, 1, (64, 3072))
output = model.forward(X, 'train')
print 'origin'
for key, value in storage.items():
  print key, np.std(value)

rescale(mlp, X, model.params)
rescaled_output = model.forward(X, 'train')
print 'rescaled'
for key, value in storage.items():
  print key, np.std(value)
print np.mean(np.abs(output - rescaled_output))
