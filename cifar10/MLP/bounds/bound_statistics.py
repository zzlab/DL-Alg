import argparse
import cPickle as pickle
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--activation')
parser.add_argument('--hidden-layers', type=int)

args = parser.parse_args()

activation = args.activation
hidden_layers = args.hidden_layers

'''
activation = 'DReLU'
hidden_layers = 3
'''
shape = (1024,) * hidden_layers + (10,)

model_path = ('models/CIFAR-%s10-%s-parameters' % ('{}-' * (len(shape) - 1), activation)).format(*shape[:-1])
parameters, _ = pickle.load(open(model_path, 'rb'))

bins = 8
histograms = []
for layer in range(hidden_layers):
  lower = parameters['lower%d' % layer].asnumpy()
  upper = parameters['upper%d' % layer].asnumpy()
  lower_histogram = np.histogram(lower, bins, density=True)
  upper_histogram = np.histogram(upper, bins, density=True)
  histograms.append((lower_histogram, upper_histogram))
path = ('models/CIFAR-%s10-%s-histograms' % ('{}-' * (len(shape) - 1), activation)).format(*shape[:-1])
pickle.dump(histograms, open(path, 'wb'))
