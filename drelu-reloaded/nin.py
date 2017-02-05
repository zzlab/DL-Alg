from mx_layers import *
def nin(activate):
  def activated_convolution(**kwargs):
    network = convolution(**kwargs)
    return activate(X=network)
  
  network = variable('data')
  network = activated_convolution(X=network, kernel_shape=(5, 5), n_filters=192, stride=(1, 1), pad=(2, 2))
  network = activated_convolution(X=network, kernel_shape=(1, 1), n_filters=160, stride=(1, 1), pad=(0, 0))
  network = activated_convolution(X=network, kernel_shape=(1, 1), n_filters=96, stride=(1, 1), pad=(0, 0))
  network = pooling(X=network, mode='maximum', kernel_shape=(3, 3), stride=(2, 2), pad=(1, 1))
  network = dropout(network, 0.5)
  network = activated_convolution(X=network, kernel_shape=(5, 5), n_filters=192, stride=(1, 1), pad=(2, 2))
  network = activated_convolution(X=network, kernel_shape=(1, 1), n_filters=192, stride=(1, 1), pad=(0, 0))
  network = activated_convolution(X=network, kernel_shape=(1, 1), n_filters=192, stride=(1, 1), pad=(0, 0))
  network = pooling(X=network, mode='average', kernel_shape=(3, 3), stride=(2, 2), pad=(1, 1))
  network = dropout(network, 0.5)
  network = activated_convolution(X=network, kernel_shape=(3, 3), n_filters=192, stride=(1, 1), pad=(1, 1))
  network = activated_convolution(X=network, kernel_shape=(1, 1), n_filters=192, stride=(1, 1), pad=(0, 0))
  network = activated_convolution(X=network, kernel_shape=(1, 1), n_filters=10, stride=(1, 1), pad=(0, 0))
  network = pooling(X=network, mode='average', kernel_shape=(8, 8), stride=(1, 1), pad=(0, 0))
  network = flatten(network)
  network = softmax_loss(network)
  return network
