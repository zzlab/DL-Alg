def output_shape(symbol, **shapes):
  _, output_shapes, _ = symbol.infer_shape(**shapes)
  return output_shapes[0]

def parameter_shapes(symbol, **shapes):
  parameters = symbol.list_arguments()
  parameter_shapes, _, _ = symbol.infer_shape(**shapes)
  return dict(zip(parameters, parameter_shapes))

def state_shapes(symbol, **shapes):
  states = symbol.list_auxiliary_states()
  _, _, state_shapes = symbol.infer_shape(**shapes)
  return dict(zip(states, state_shapes))

def mxnet_array_mapping(mapping, context=None):
  import mxnet as mx
  if context is None: context = mx.cpu()
  return {key : mx.nd.array(value, context) for key, value in mapping.items()}
