def convert_labels(labels,classes=10):
  import minpy.numpy as np
  converted=np.zeros((labels.shape[0],classes))
  converted[np.arange(labels.shape[0]),labels]=1
  return converted

def load_mnist(path='.', shape=None):
  import cPickle,gzip
  import minpy.numpy as np
  with gzip.open(path+'/mnist.gz', 'rb') as data:
    package = cPickle.load(data)
  if shape is not None:
    package = list(package) 
    for index, data in enumerate(package):
      X, Y = data
      N, D = X.shape
      package[index] = (X.reshape((N,) + shape), Y)
    package = tuple(package)
  unpacked = []
  for data in package:
    unpacked.extend(data)
  unpacked = tuple(unpacked)
  return unpacked

if __name__ == '__main__':
  data = load_mnist(shape=(7, 4 * 28))
  for d in data:
    print d.min(), d.max()
