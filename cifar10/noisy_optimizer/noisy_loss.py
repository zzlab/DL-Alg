from minpy.core import grad_and_loss as _gradient_loss
import minpy.numpy as np
from facility import KL

def unbiased_noisy_gradient_loss(model, X, Y, sample, gamma, K, N_CLASSES=4):
  N, D = X.shape
  noisy_X = sample(K, D)
  def _loss_function(*args):
    normal_loss = model.loss(model.forward(X, 'train'), Y)
    noisy_output = model.forward(noisy_X, 'train')
    noisy_output -= np.max(noisy_output, axis=1).reshape((K, 1))
    noisy_output = np.exp(noisy_output)
    model_p_noisy_X = noisy_output / np.sum(noisy_output, axis=1).reshape((K, 1))
    kl = KL(1.0 / N_CLASSES, model_p_noisy_X)
    noisy_loss = gamma * np.sum(kl) / float(K)
    return gamma * normal_loss + (1 - gamma) * noisy_loss
  gl = _gradient_loss(_loss_function, range(len(model.params)))
  parameters = list(model.params.values())
  return gl(*parameters)

import mxnet.symbol as symbols
from mx_layers import *
from mx_models.mlp import MLP

class NoisyMLP:
  # TODO weight sharing
  def __call__(self, X_shape, generator, gconfigurations):
    normal_X = variable('data')
    gconfigurations['shape'] = X_shape
    noisy_X = getattr(symbols, generator)(**gconfigurations)

    for index, shape in enumerate(self._shape[:-1]):
      weight = variable('fullyconnected%d_weight' % index)
      bias = variable('fullyconnected%d_bias' % index)
      normal_X = fully_connected(normal_X, shape, weight=weight, bias=bias)
      normal_X = activate(normal_X, self._activation, data_shape),
      noisy_X = fully_connected(noisy_X, shape, weight=weight, bias=bias)
      noisy_X = activate(noisy_X, self._activation, data_shape),
    
    weight = variable('fullyconnected%d_weight' % (len(self._shape) - 1))
    bias = variable('fullyconnected%d_bias' % (len(self._shape) - 1))
    normal_X = fully_connected(normal_X, self._shape[-1], weight=weight, bias=bias)
    noisy_X = fully_connected(noisy_X, self._shape[-1], weight=weight, bias=bias)
    
    normal_loss = softmax_loss(normal_X)
    noisy_p = softmax_activation(noisy_X)
    noisy_loss = NoisyMLP._kl(noisy_p, self._shape[-1])

    return normal_loss + noisy_loss

  def __init__(self, shape, activation):
    self._shape      = shape
    self._activation = activation

  @staticmethod
  def _kl(P, n_classes):
    return symbols.sum(symbols.log(P), axis=1) / float(n_classes)
