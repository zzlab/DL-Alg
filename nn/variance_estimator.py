def deviation(limit):
  import math
  loss = (2 / math.pi) ** 0.5 * limit * math.exp(- limit ** 2 / 2) - (limit ** 2 - 1) * math.erfc(limit / 2 ** 0.5)
  return (1 - loss) ** 0.5

if __name__ == '__main__':
  import sys
  limit = float(sys.argv[1])
  print deviation(limit)
