from MXLayers import *

class NIN:
  def __init__(self, activation, initializer):
    self.activation  = activation
    self.initializer = initializer
   
  def __call__(self, data_shape):
    def activated_convolution(*args, **kwargs):
      return convolution(*args, activation=self.activation, data_shape=data_shape, **kwargs)

    inputs = mx.sym.Variable('data')
    c0 = activated_convolution(inputs, (5, 5), 192, pad=(2, 2))
    c1 = activated_convolution(c0, (1, 1), 160)
    c2 = activated_convolution(c1, (1, 1), 96)
    p0 = pooling(c2, 'max', (3, 3), (2, 2), (1, 1))
    d0 = dropout(p0, 0.5)
    c3 = activated_convolution(d0, (5, 5), 192, pad=(2, 2))
    c4 = activated_convolution(c3, (1, 1), 192)
    c5 = activated_convolution(c4, (1, 1), 192)
    p1 = pooling(c5, 'avg', (3, 3), (2, 2), (1, 1))
    d1 = dropout(p1, 0.5)
    c6 = activated_convolution(d1, (3, 3), 192, pad=(1, 1))
    c7 = activated_convolution(c6, (1, 1), 192)
    c8 = activated_convolution(c7, (1, 1), 10)
    p2 = pooling(c8, 'avg', (8, 8), (1, 1))
    outputs = softmax(reshape(p2, (0, 10)))
    return outputs, self.initializer
