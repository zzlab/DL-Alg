from minpy.core import grad_and_loss as _gradient_loss

def initialize(model):
  import minpy.numpy as np
  import minpy.nn.init as initializers
  from minpy.context import cpu
  for key, value in model.param_configs.items():
    model.params[key] = getattr(
      initializers,
      value['init_rule']
    ) (
      value['shape'],
      value.get('init_config', {})
    )
    '''
    model.params[key] =  getattr(
      initializers,
      'gaussian'
    ) (
      value['shape'],
      {'stdvar' : 0.01}
    )
    print key, 'initialized', value['init_rule'], value.get('init_config', {})
    '''

def gradient_loss(model, X, Y):
  def _loss_function(X, Y, *args):
    predictions = model.forward(X, 'train')
    return model.loss(predictions, Y)

  gl = _gradient_loss(_loss_function, range(2, len(model.params) + 2))
  parameters = list(model.params.values())
  return gl(X, Y, *parameters)

class Updater:
  def __init__(self, model, optimizer, settings, policy=None):
    import minpy.nn.optim as optimizers
    self._optimizer = getattr(optimizers, optimizer)
    self._parameters = model.params
    self._parameter_keys = list(self._parameters.keys())
    self._parameter_values = list(self._parameters.values())
    self._cache = {key : dict(settings) for key in self._parameter_keys}
    self._policy = policy

  def __setitem__(self, key, value):
    for cache in self._cache.values():
      cache[key] = value

  def update(self, gradients):
    mapped_gradients = dict(zip(self._parameter_keys, gradients))
    for key, value in mapped_gradients.items():
      self._parameters[key], self._cache[key] = self._optimizer(
        self._parameters[key],
        value,
        self._cache[key]
      )

def classification_accuracy(model, X, Y):
  import minpy.numpy as np
  N = X.shape[0]
  predictions = model.forward(X, 'test')
  predicted_Y = np.argmax(predictions, axis=1)
  errors = np.count_nonzero(predicted_Y - Y)
  return 1 - errors / float(N)

def Batches(data, batch_size):
  n_batches = data.shape[0] // batch_size
  index = 0
  while True:
    if index + 1 == n_batches:
      yield data[index * batch_size:]
      index = 0
    else:
      yield data[index * batch_size : (index + 1) * batch_size]
      index += 1
