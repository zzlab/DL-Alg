def load_mnist(path=None, scale=None, shape=None):
  import cPickle, gzip, joblib, os
  import numpy as np

  if path is None: path = os.environ['MNISTPATH']
  package = joblib.load(path + '/mnist')
  '''
  with gzip.open(path + '/mnist.gz', 'rb') as data:
    package = cPickle.load(data)
  '''

  unpacked = []

  for data in package:
    X, Y = data
    X = X.astype(np.float32)
    if scale is not None:
      X = X * scale / X.max()
    if shape is not None:
      N, D = X.shape
      X = X.reshape((N,) + shape)
    unpacked.extend((X, Y))

  return tuple(unpacked)

def load_cifar10(path=None, reshape=False, center=False, rescale=False, validation=4):
  import cPickle as pickle
  import os
  import numpy as np
  if path is None: path = os.environ['CIFAR10PATH']
  training_data, training_labels = [], []
  for i in range(5):
    if i != validation:
      batch = pickle.load(open(path + 'data_batch_%d' % (i + 1), 'rb'))
      training_data.append(batch['data'])
      training_labels.append(np.array(batch['labels']))
  training_data = np.vstack(training_data)
  if center:
    mean = training_data.mean(axis=0).reshape((1, 3072))
    training_data = training_data - mean
  if rescale:
    deviation = (((training_data ** 2).mean(axis=0)) ** 0.5).reshape((1, 3072))
    training_data = training_data / deviation
  if reshape: training_data = training_data.reshape((40000, 3, 32, 32))
  training_labels = np.hstack(training_labels)

  if validation is None: validation_batch = pickle.load(open(path + 'test_batch', 'rb'))
  else: validation_batch = pickle.load(open(path + 'data_batch_%d' % (validation + 1), 'rb'))
  validation_data = np.array(validation_batch['data'])
  if center: validation_data = validation_data - mean
  if rescale: validation_data = validation_data / deviation
  if reshape: validation_data = validation_data.reshape((10000, 3, 32, 32))
  validation_labels = np.array(validation_batch['labels'])

  test_batch = pickle.load(open(path + 'test_batch', 'rb'))
  test_data = np.array(test_batch['data'])

  if center: test_data = test_data - mean
  if rescale: test_data = test_data / deviation
  if reshape: test_data = test_data.reshape((10000, 3, 32, 32))
  test_labels = np.array(test_batch['labels']) 
  
  return \
    training_data, training_labels, \
    validation_data, validation_labels, \
    test_data, test_labels

def load_cifar10_record(batch_size=None, path=None):
  import os
  from mxnet.io import ImageRecordIter
  if path is None: path = os.environ['CIFAR10PATH']
  training_record = '%s/training-record' % path
  validation_record = '%s/validation-record' % path

  r_mean = 123.680
  g_mean = 116.779
  b_mean = 103.939
  mean = int(sum((r_mean, g_mean, b_mean)) / 3)
  scale = 1 / 59.4415

  training_data = ImageRecordIter(
    batch_size         = batch_size,
    data_name          = 'data',
    data_shape         = (3, 32, 32),
    fill_value         = mean,
    label_name         = 'softmax_label',
    label_width        = 1,
    mean_r             = r_mean,
    mean_g             = g_mean,
    mean_b             = b_mean,
    pad                = 4,
    path_imgrec        = training_record,
    preprocess_threads = 16,
    rand_crop          = True,
    rand_mirror        = True,
    scale              = scale,
    shuffle            = True,
    verbose            = False,
  )
  validation_data = ImageRecordIter(
    batch_size         = batch_size,
    data_name          = 'data',
    data_shape         = (3, 32, 32),
    label_name         = 'softmax_label',
    label_width        = 1,
    mean_r             = r_mean,
    mean_g             = g_mean,
    mean_b             = b_mean,
    num_parts          = 2,
    part_index         = 0,
    path_imgrec        = validation_record,
    preprocess_threads = 16,
    scale              = scale,
    verbose            = False,
  )
  test_data = ImageRecordIter(
    batch_size         = batch_size,
    data_name          = 'data',
    data_shape         = (3, 32, 32),
    label_name         = 'softmax_label',
    label_width        = 1,
    mean_r             = r_mean,
    mean_g             = g_mean,
    mean_b             = b_mean,
    num_parts          = 2,
    part_index         = 1,
    path_imgrec        = validation_record,
    preprocess_threads = 16,
    scale              = scale,
    verbose            = False,
  )

  return training_data, validation_data, test_data # TODO validation/test distinction

def load_imagenet_record(batch_size=None, path=None):
  import os
  from mxnet.io import ImageRecordIter
  if path is None: path = os.environ['IMAGENETPATH']
  training_record = '%s/training.record' % path
  validation_record = '%s/validation.record' % path

  r_mean = 123.680
  g_mean = 116.779
  b_mean = 103.939
  mean = int((r_mean + g_mean + b_mean) / 3)
  scale = 1 / 59.4415

  training_data = ImageRecordIter(
    batch_size         = batch_size,
    data_name          = 'data',
    data_shape         = (3, 224, 224),
    fill_value         = mean,
    label_name         = 'softmax_label',
    label_width        = 1,
    mean_r             = r_mean,
    mean_g             = g_mean,
    mean_b             = b_mean,
    pad                = 4,
    path_imgrec        = training_record,
    preprocess_threads = 16,
    rand_crop          = True,
    rand_mirror        = True,
    scale              = scale,
    shuffle            = True,
    verbose            = False,
  )

  validation_data = ImageRecordIter(
    batch_size         = batch_size,
    data_name          = 'data',
    data_shape         = (3, 224, 224),
    fill_value         = mean,
    label_name         = 'softmax_label',
    label_width        = 1,
    mean_r             = r_mean,
    mean_g             = g_mean,
    mean_b             = b_mean,
    pad                = 4,
    path_imgrec        = validation_record,
    preprocess_threads = 16,
    rand_crop          = True,
    rand_mirror        = True,
    scale              = scale,
    shuffle            = True,
    verbose            = False,
  )

  return training_data, validation_data

if __name__ == '__main__':
  training, validation = load_imagenet_record(batch_size=1)
