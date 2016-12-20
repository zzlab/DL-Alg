import cPickle as pickle
import time

def convert_labels(labels,classes=10):
  import numpy as np
  converted=np.zeros((labels.shape[0],classes))
  converted[np.arange(labels.shape[0]),labels]=1
  return converted

def load_whitened_cifar10(path='whitened-cifar/', reshape=False):
  import numpy as np
  training_X, training_Y = [], []
  for i in range(4):
    batch = pickle.load(open('%s/training%d' % (path, i), 'rb'))
    if reshape:
      batch['data'] = batch['data'].reshape((len(batch['data']), 3, 32, 32))
    training_X.append(batch['data'])
    training_Y.append(batch['labels'])
  training_X = np.vstack(training_X)
  training_Y = np.hstack(training_Y)

  validation_batch = pickle.load(open('%s/validation' % path, 'rb'))
  if reshape:
    validation_batch['data'] = \
      validation_batch['data'].reshape((len(validation_batch['data']), 3, 32, 32))
  validation_X, validation_Y = validation_batch['data'], validation_batch['labels']

  test_batch = pickle.load(open('%s/test' % path, 'rb'))
  if reshape:
    test_batch['data'] = \
      test_batch['data'].reshape((len(test_batch['data']), 3, 32, 32))
  test_X, test_Y = test_batch['data'], test_batch['labels']

  return training_X, training_Y, validation_X, validation_Y, test_X, test_Y
  
def load_cifar10(path='./cifar/', reshape=False, convert=False, center=False, rescale=False, validation=4):
  import numpy as np
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

  if reshape:
    training_data = training_data.reshape((40000, 3, 32, 32))

  training_labels = convert_labels(np.hstack(training_labels)) if convert else np.hstack(training_labels)

  if validation is None:
    validation_batch = pickle.load(open(path + 'test_batch', 'rb'))
  else:
    validation_batch = pickle.load(open(path + 'data_batch_%d' % (validation + 1), 'rb'))

  validation_data = np.array(validation_batch['data'])

  if center:
    validation_data = validation_data - mean
  if rescale:
    validation_data = validation_data / deviation

  if reshape:
    validation_data = validation_data.reshape((10000, 3, 32, 32))

  validation_labels = np.array(validation_batch['labels'])

  test_batch = pickle.load(open(path + 'test_batch', 'rb'))
  test_data = np.array(test_batch['data'])

  if center:
    test_data = test_data - mean
  if rescale:
    test_data = test_data / deviation

  if reshape:
    test_data = test_data.reshape((10000, 3, 32, 32))
  test_labels = np.array(test_batch['labels']) 
  
  return training_data, training_labels, \
    validation_data, validation_labels, \
    test_data, test_labels

def load_modified_cifar10(path='./cifar', reshape=False):
  training   = [pickle.load(open('%s/training_%d' % (path, i), 'rb')) for i in range(10)]
  validation = pickle.load(open('%s/validation' % path, 'rb'))
  test       = pickle.load(open('%s/test' % path, 'rb'))
  statistics = pickle.load(open('%s/statistics' % path, 'rb'))

  for i in range(10):
    training[i]['data'] = training[i]['data'] - statistics['mean']
    training[i]['data'] /= statistics['deviation']
   
  validation['data'] = validation['data'] - statistics['mean']
  validation['data'] /= statistics['deviation']

  test['data'] = test['data'] - statistics['mean']
  test['data'] /= statistics['deviation']

  if reshape:
    for i in range(10):
      training[i]['data']   = training[i]['data'].reshape((len(training[i]['data']), 3, 32, 32))
    validation['data'] = validation['data'].reshape((len(validation['data']), 3, 32, 32))
    test['data']       = test['data'].reshape((len(test['data']), 3, 32, 32))
  return training, validation, test

if __name__ == '__main__':
  import time
  data = load_whitened_cifar10()
