import mxnet as mx
import math
import numpy as np
import re
import logging

class MXSolver():
  def __init__(self, model, **kwargs):
    self.batch_size      = kwargs['batch_size']
    self.data            = kwargs['data']
    self.devices         = [mx.gpu(index) for index in kwargs['devices']]
    self.epoch           = kwargs['epoch']
    self.metric          = kwargs.pop('metric', mx.metric.Accuracy())

    self.optimizer_settings = kwargs.pop('optimizer_settings', {})
    if self.optimizer_settings['optimizer'] == 'SGD':
      self.optimizer_settings['optimizer'] = 'ccSGD'
    __optimizer_settings = {key : value for key, value in self.optimizer_settings.items()}
    batch_count = math.ceil(self.data[0].shape[0] / self.batch_size)
    self.optimizer = getattr(
      mx.optimizer,
      self.optimizer_settings.pop('optimizer')
    )(
      learning_rate = self.optimizer_settings.pop('lr'),
      lr_scheduler  = mx.lr_scheduler.FactorScheduler(
        step   = self.optimizer_settings.pop('lr_decay_interval') * batch_count,
        factor = self.optimizer_settings.pop('lr_decay_factor')
      ),
      rescale_grad = (1.0 / self.batch_size),
      wd           = self.optimizer_settings.pop('weight_decay'),
      **self.optimizer_settings
    )
    self.optimizer_settings = __optimizer_settings

    batch_shape = (self.batch_size,) + self.data[0].shape[1:]

    symbol, initializer = model(batch_shape)
    self.model = mx.model.FeedForward(
      ctx           = self.devices,
      initializer   = initializer,
      num_epoch     = self.epoch,
      optimizer     = self.optimizer,
      symbol        = symbol
    )

    print 'model constructed'

  def train(self):
    progress = { \
      'epoch'    : 0, \
      'accuracy' : [0] \
    }

    class AccuracyFilter(logging.Filter):
      def __init__(self, model):
        self.model = model
        self.memory_allocated = False
        self.arg_params, self.aux_params = None, None
      def filter(self, record):
        message = record.getMessage()
        if 'Validation' in message:
          history = progress['accuracy']
          history.append(float(re.search('\d\.\d+', message).group()))
          print 'epoch {:<3} accuracy {}'.format(progress['epoch'], history[-1])
          if history[-1] > max(history[:-1]):
            if not self.memory_allocated:
              self.arg_params = {
                key : mx.nd.zeros(value.shape, mx.cpu()) for key, value in self.model.arg_params.items()
              }
              self.aux_params = {
                key : mx.nd.zeros(value.shape, mx.cpu()) for key, value in self.model.aux_params.items()
              }
              self.memory_allocated = True
            for key, value in self.model.arg_params.items():
              value.copyto(self.arg_params[key])
            for key, value in self.model.aux_params.items():
              value.copyto(self.aux_params[key])
            print 'epoch {:<3} checkpointed'.format(progress['epoch'])
          return False
        else:
          return True

    class BlockFilter(logging.Filter):
      def __init__(self):
        self.targets = (
          'Auto-select kvstore type',
          'Start training with',
          'Resetting Data Iterator'
        )
      def filter(self, record):
        message = record.getMessage()
        if 'Epoch' in message:
          progress['epoch'] = int(re.search('Epoch\[(\d+)\]', message).group(1)) + 1
        return all(target not in message for target in self.targets)

    class LRDecayFilter(logging.Filter):
      def __init__(self, settings):
        self.optimizer_settings = settings
      def filter(self, record):
        message = record.getMessage()
        if 'Change learning rate to' in message:
          lr = float(message.split(' ')[-1])
          if lr != self.optimizer_settings['lr']:
            print 'learning rate decayed to %f' % lr
          return False
        else:
          return True

    class TimeFilter(logging.Filter):
      def filter(self, record):
        message = record.getMessage()
        if 'Time cost' in message:
          time_consumed = float(message.split('=')[-1])
          print 'epoch {:<3} {} seconds consumed'.format(progress['epoch'], time_consumed)
          return False
        else:
          return True

    logging.basicConfig(level=logging.NOTSET)
    logger = logging.getLogger()
    accuracy_filter = AccuracyFilter(self.model)
    logger.addFilter(accuracy_filter)
    logger.addFilter(BlockFilter())
    logger.addFilter(LRDecayFilter(self.optimizer_settings))
    logger.addFilter(TimeFilter())

    class EpochCallback:
      def __call__(self, *args):
        return

    self.model.fit(
      X                  = self.data[0],
      y                  = self.data[1],
      eval_data          = (self.data[2], self.data[3]),
      eval_metric        = self.metric,
      epoch_end_callback = EpochCallback(),
      logger             = logger
    )

    '''
    arg_params, aux_params = accuracy_filter.arg_params, accuracy_filter.aux_params
    for key in self.model.arg_params:
      arg_params[key].copyto(self.model.arg_params[key])
    for key in self.model.aux_params:
      aux_params[key].copyto(self.model.aux_params[key])
    test_data = mx.io.NDArrayIter(self.data[4], batch_size=self.data[4].shape[0])
    score = self.model.predict(test_data)
    prediction = score.argmax(axis=1)
    test_accuracy = 1 - np.count_nonzero(prediction - self.data[5]) / float(self.data[5].shape[0])
    print 'optimal validation accuracy %f (epoch %d)' % (
      max(progress['accuracy']),
      np.array(progress['accuracy']).argmax()
    )
    print 'test accuracy %f' % (test_accuracy)
    '''

    return progress['accuracy'] # , test_accuracy
