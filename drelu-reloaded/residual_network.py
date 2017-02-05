import mxnet as mx
from mxnet.symbol import Convolution
from mx_layers import *

def _normalized_convolution(X, kernel_shape, n_filters, stride, pad, activate):
  network = convolution(X=X, kernel_shape=kernel_shape, n_filters=n_filters, stride=stride, pad=pad)
  network = batch_normalization(network, fix_gamma=False)
  network = activate(network)
  return network

_WIDTH, _HEIGHT = 3, 3

def _transit(X, n_filters, activate):
  P = _normalized_convolution(X, (_WIDTH, _HEIGHT), n_filters, (2, 2), (1, 1), activate)
  P = _normalized_convolution(P, (_WIDTH, _HEIGHT), n_filters, (1, 1), (1, 1), activate)
  Q = pooling(X=X, mode='average', kernel_shape=(2, 2), stride=(2, 2), pad=(0, 0))
  pad_width = (0, 0, 0, n_filters / 2, 0, 0, 0, 0)
  Q = pad(Q, pad_width, 'constant')
  return P + Q

def _recur(X, n_filters, activate, times):
  residual = _normalized_convolution(X, (_WIDTH, _HEIGHT), n_filters, (1, 1), (1, 1), activate)
  for t in range(times - 1):
    residual = _normalized_convolution(residual, (_WIDTH, _HEIGHT), n_filters, (1, 1), (1, 1), activate)
  return X + residual

def residual_network(n, activate, times):
  global _WIDTH, _HEIGHT
  network = variable('data')

  FILTER_IN, FILTER_OUT = 16, 16
  network = _normalized_convolution(network, (_WIDTH, _HEIGHT), FILTER_IN, (1, 1), (1, 1), activate)
  for i in range(n): network = _recur(network, FILTER_OUT, activate, times)

  FILTER_IN, FILTER_OUT = 32, 32
  network = _transit(network, FILTER_IN, activate)
  for i in range(n - 1): network = _recur(network, FILTER_OUT, activate, times)

  FILTER_IN, FILTER_OUT = 64, 64
  network = _transit(network, FILTER_IN, activate)
  for i in range(n - 1): network = _recur(network, FILTER_OUT, activate, times)

  network = pooling(X=network, mode='average', kernel_shape=(8, 8), stride=(1, 1), pad=(0, 0))
  network = flatten(network)
  network = fully_connected(X=network, n_hidden_units=10)
  network = softmax_loss(network, normalization='batch')
  return network
