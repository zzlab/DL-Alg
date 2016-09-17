def convert_labels(labels,classes=10):
  import minpy.numpy as np
  converted=np.zeros((labels.shape[0],classes))
  converted[np.arange(labels.shape[0]),labels]=1
  return converted

def load_mnist(path='.'):
  import cPickle,gzip
  import minpy.numpy as np
  with gzip.open(path+'/mnist.gz', 'rb') as data:
    training,validation,test=cPickle.load(data)
    return training[0], training[1], validation[0], validation[1], test[0], test[1]

if __name__ == '__main__':
  data = load_mnist(origin=True)
  for d in data:
    print d.shape
