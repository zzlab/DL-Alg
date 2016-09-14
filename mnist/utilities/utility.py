import minpy.numpy as np

import sys
sys.path.append('../..')

def accuracy(p,l):
  if len(l.shape) == 1:
    return 1-np.count_nonzero(p-l).val/float(p.shape[0])
  else:
    inputs, labels = p, l
    return np.mean(np.sum((inputs-labels)**2, axis=1))

def sparsity(l):
  return 1-np.count_nonzero(l).val/float(l.asnumpy().size)

def save_model(model, path):
  import pickle
  model.parameters=[p.asnumpy() for p in model.parameters]
  pickle.dump(model,open(path,'wb'))

def load_model(path):
  import pickle
  model=pickle.load(open(path,'rb'))
  model.parameters=[np.copy(p) for p in model.parameters]
  return model

def forward_layer(inputs, args):
  if isinstance(args[1], dict):
    return getattr(__import__('layers'), args[0])(inputs, **args[1])
  else:  
    return getattr(__import__('layers'), args[0])(inputs, *args[1:])

def norm(matrix, p=2):
  return np.sum(matrix ** p) ** (1 / float(p))

def sigmoid(X):
  return 1/(1+np.exp(-X))

def PDF_to_CDF(PDF):
  assert sum(PDF[1]) == 1
  return (PDF[0], [sum(PDF[1][:i]) for i in range(1, len(PDF[1])+1)])

def sample(CDF):
  import random
  r = random.random()
  for i in range(len(CDF[1])):
    if r < CDF[1][i]:
      return CDF[0][i]
      
#if __name__ == '__main__':
