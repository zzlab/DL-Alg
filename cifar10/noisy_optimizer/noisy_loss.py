def noisy_gradient_loss(model, X, Y, gamma, K, N_CLASSES=4):
  N, D = X.shape
  noisy_X = sample(K, D)
  p_noisy_X = pdf(noisy_X)
  def _loss_function(*args):
    normal_loss = model.loss(model.forward(X, 'train'), Y)
    noisy_output = model.forward(noisy_X, 'train')
    noisy_output -= np.max(noisy_output, axis=1).reshape((K, 1))
    noisy_output = np.exp(noisy_output)
    model_p_noisy_X = noisy_output / np.sum(noisy_output, axis=1).reshape((K, 1))
    kl = KL(1.0 / N_CLASSES, model_p_noisy_X)
    noisy_loss = gamma * np.sum(kl) / float(K)
    return normal_loss + noisy_loss
  gl = _gradient_loss(_loss_function, range(len(model.params)))
  parameters = list(model.params.values())
  return gl(*parameters)

import mxnet.symbol as symbols
from mx_layers import *
from mx_models.mlp import MLP

class NoisyMLP:
  def __call__(self, X_shape, generator, gconfigurations):
    normal_X = variable('data')
    normal_loss = softmax_loss(self._generate_network(normal_X, X_shape))
    gconfigurations['shape'] = X_shape
    noisy_X = getattr(symbols, generator)(**gconfigurations)
    noisy_p = softmax_activation(self._generate_network(noisy_X, X_shape))
    noisy_loss = NoisyMLP._kl(noisy_p, self._shape[-1])
    return normal_loss + noisy_loss

  def __init__(self, shape, activation):
    self._shape      = shape
    self._activation = activation

  def _generate_network(self, X, data_shape):
    network = reduce(
      lambda symbol, d : activate(fully_connected(symbol, d), self._activation, data_shape),
      self._shape[:-1],
      X
    )
    return fully_connected(network, self._shape[-1])

  @staticmethod
  def _kl(P, n_classes):
    return symbols.sum(symbols.log(P), axis=1) / float(n_classes)
