from operator import mul

def receptive_field(convolutions):
  kernels, strides = zip(*convolutions)
  results = [1]
  for index, kernel in enumerate(kernels):
    field = results[-1] + (kernel - 1) * reduce(mul, strides[:index], 1)
    results.append(field)
  return results

if __name__ is '__main__':
  convolutions = (
      (5, 1),
      (2, 2),
      (5, 1),
      (2, 2),
      (5, 1),
  )
  print receptive_field(convolutions)
