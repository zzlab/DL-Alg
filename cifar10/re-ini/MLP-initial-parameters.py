import cPickle as pickle
import sys

import minpy.numpy as np
import minpy.nn.model_builder as builder

sys.path.append('../../nn')
from solver_primitives import *

ACTIVATION = sys.argv[1]
ACTIVATION = ACTIVATION.replace('BN', '')
BN = 'BN' in sys.argv[1]
activation = getattr(builder, ACTIVATION)
shapes = [int(shape) for shape in sys.argv[2:]]

mlp = builder.Sequential()
for i, shape in enumerate(shapes[:-1]):
  mlp.append(builder.Affine(shape))
  print shape
  if BN:
    mlp.append(builder.BatchNormalization())
  mlp.append(activation())
mlp.append(builder.Affine(shapes[-1]))
print shapes[-1]
model = builder.Model(mlp, 'softmax', (3072,))

initialize(model)

'''
X = np.random.normal(0, 0.1, (1024, 3072))
Y = np.random.choice(np.arange(10), 1024)
output = model.forward(X, 'train')
loss = model.loss(output, Y)
print loss
'''

p = {key : value.asnumpy() for key, value in model.params.items()}
for key, value in p.items():
  if 'weight' in key:
    p[key] = value.T
  if 'beta' in key:
    mean = key.replace('beta', 'moving_mean')
    p[mean] = np.zeros(value.shape).asnumpy()
  if 'gamma' in key:
    variance = key.replace('gamma', 'moving_var')
    p[variance] = np.zeros(value.shape).asnumpy()
    
configuration = '-'.join(sys.argv[1:])
pickle.dump(p, open('model/MLP-initial-%s' % configuration, 'wb'))
