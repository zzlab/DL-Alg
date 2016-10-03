def std(limit):
  import math
  loss = (2 / math.pi) ** 0.5 * limit * math.exp(- limit ** 2 / 2.0) - (limit ** 2 - 1) * math.erfc(limit / 2.0 ** 0.5)
  return (1 - loss) ** 0.5

def infer_std(inputs, weight_shape):
  import minpy.numpy as np
  E_X_2 = np.mean(inputs ** 2)
  return 1 / (E_X_2 * float(weight_shape[0])) ** 0.5

if __name__ == '__main__':
  import sys
  r = float(sys.argv[1].replace('*', '-'))
  s = float(sys.argv[2].replace('*', '-')) if len(sys.argv) == 3 else None
  print std(r)
