import minpy.numpy as np
import minpy.nn.model_builder as builder
from minpy.core import grad_and_loss as _gradient_loss

from minpy.context import set_context, gpu, cpu
set_context(gpu(0))

import numpy as np0
from scipy.stats import multivariate_normal as gaussian
from scipy.stats import uniform

import sys
sys.path.append('../../nn/')
from facility import *
from solver_primitives import *

def generate_data(N, D, mean=0, std=1):
  mean = np0.full(D, mean)
  covariance_matrix = np0.eye(D) * std
  data = np0.random.multivariate_normal(mean, covariance_matrix, N)
  p = gaussian.pdf(data, mean, covariance_matrix)
  return data, p

def gan_gradient_loss(dmodel, gmodel, X, delta=0.1):
  N, D = X.shape
  noise = np.random.uniform(np_min(X), np_max(X), X.shape)
  lower, upper = delta, 1 - delta

  def gan_loss(*args):
    p_X = dmodel.forward(X, 'train')
    random_X = gmodel.forward(noise, 'train')
    p_random_X = dmodel.forward(random_X, 'train')
    value = np.log(clip(p_X, lower, upper)) + np.log(clip(1 - p_random_X, lower, upper))
    loss = np.sum(value) / float(N)
    return loss

  gl = _gradient_loss(gan_loss, range(len(dmodel.params) + len(gmodel.params)))
  parameters = list(dmodel.params.values()) + list(gmodel.params.values())
  return gl(*parameters)
  
N, D = 50000, 16
data, p = generate_data(N, D)
BATCH_SIZE = 100
X_batches = Batches(data, BATCH_SIZE)
p_batches = Batches(p.reshape((N, 1)), BATCH_SIZE)

ACTIVATION = 'ReLU'
activation = getattr(builder, ACTIVATION)

DSHAPE = (16,) * 4 + (1,)
dmlp = builder.Sequential()
for shape in DSHAPE[:-1]:
  dmlp.append(builder.Affine(shape))
  dmlp.append(activation())
dmlp.append(builder.Affine(DSHAPE[-1]))
dmodel = builder.Model(dmlp, 'l2', (D,))
initialize(dmodel)
dupdater = Updater(dmodel, 'sgd', {'learning_rate' : -0.01})

GSHAPE = (16,) * 4 + (D,)
gmlp = builder.Sequential()
for shape in GSHAPE[:-1]:
  gmlp.append(builder.Affine(shape))
  gmlp.append(activation())
gmlp.append(builder.Affine(GSHAPE[-1]))
gmodel = builder.Model(gmlp, 'l2', (D,))
initialize(gmodel)
gupdater = Updater(gmodel, 'sgd', {'learning_rate' : 0.01})

ITERATIONS = 1000
INTERVAL = 10

for i in range(ITERATIONS):
  X_batch = next(X_batches)
  p_batch = next(p_batches)
  gradients, loss = gan_gradient_loss(dmodel, gmodel, X_batch)

  offset = len(dmodel.params)
  dupdater.update(gradients[:offset])
  gupdater.update(gradients[offset:])

  loss = to_float(loss)
  if (i + 1) % INTERVAL == 0:
    print 'iteration %d loss %f' % (i + 1, loss)
