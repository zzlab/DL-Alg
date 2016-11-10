from minpy.core import grad_and_loss as _gradient_loss
import minpy.numpy as np

def stochastic_gradient_loss(model, X, Y, sigma):
  mean = np.mean(X)
  noise_X = np.random.normal(mean, sigma, X.shape)
  def _loss_function(*args):
    predictions = model.forward(X, 'train')
    noisy_predictions = model.forward(noise_X, 'train')
    return model.loss(predictions, Y) + model.loss(noisy_predictions, Y)

  gl = _gradient_loss(_loss_function, range(len(model.params)))
  parameters = list(model.params.values())
  return gl(*parameters)

def baseline_gradient_loss(model, X, Y, sigma):
  noise_X = np.random.normal(mean, sigma, X.shape)
  def _loss_function(*args):
    noisy_predictions = model.forward(noise_X, 'train')
    return model.loss(noisy_predictions, Y)

  gl = _gradient_loss(_loss_function, range(len(model.params)))
  parameters = list(model.params.values())
  return gl(*parameters)
