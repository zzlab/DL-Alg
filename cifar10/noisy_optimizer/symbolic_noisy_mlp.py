import cPickle as pickle
import mxnet as mx
from mxnet.optimizer import SGD
from mxnet.metric import Accuracy
import minpy.nn.model_builder as builder
from noisy_loss import NoisyMLP
from solver_primitives import Batches, initialize
from utilities.data_utility import load_cifar10

from minpy.context import set_context, gpu
set_context(gpu(0))

ACTIVATION = 'ReLU'
SHAPE = 1024, 10
BATCH_SIZE = 256
noisy_mlp = NoisyMLP(SHAPE, ACTIVATION)
symbol = noisy_mlp((BATCH_SIZE, 3072), 'uniform', {'low' : -1, 'high' : 1})

X_SHAPE = (3072,)
path = 'mlp-%s-%s' % (ACTIVATION, '-'.join(str(shape) for shape in SHAPE))
arg_path = '%s-args' % path
try:
  args = pickle.load(open(arg_path, 'rb'))
except:
  activation = getattr(builder, ACTIVATION)
  mlp = builder.Sequential()
  for shape in SHAPE[:-1]:
    mlp.append(builder.Affine(shape))
    mlp.append(activation())
  mlp.append(builder.Affine(SHAPE[-1]))
  model = builder.Model(mlp, 'softmax', X_SHAPE)
  initialize(model)
  args = {key : value.asnumpy() for key, value in model.params.items()}
  pickle.dump(args, open(arg_path, 'wb'))

args = {key : mx.nd.array(value) for key, value in args.items()}
training_X, training_Y, _, _, _, _, = \
  map(mx.nd.array, load_cifar10(path='../../cifar10/utilities/cifar/', center=True, rescale=True))
args['data'] = training_X[:BATCH_SIZE]
args['softmax_label'] = training_Y[:BATCH_SIZE]

print symbol.list_arguments()
executor = symbol.bind(mx.gpu(0), args)
optimizer = SGD()
training_accuracy = Accuracy()
validation_accuracy = Accuracy()

X_batches = Batches(training_X, BATCH_SIZE)
Y_batches = Batches(training_Y, BATCH_SIZE)

ITERATIONS = 10000
LOGGING_INTERVAL = 10
VALIDATION_INTERVAL = 500
training_accuracy.reset()
loss_table = []
ta_table = []
va_table = []
for i in range(ITERATIONS):
  X_batch = next(X_batches)
  Y_batch = next(Y_batches)
  executor.forward(data=X_batch, softmax_label=Y_batch)
  training_accuracy.update(Y_batch, executor.outputs)
  loss = executor.outputs[0][0]
  loss_table.append(loss)
  training_accuracy.update(Y_batch, executor.outputs)
  if (i + 1) % LOGGING_INTERVAL == 0:
    print 'iteration %d loss %f accuracy %f' % (i + 1, loss, training_accuracy.get())
    training_accuracy.reset()

  executor.backward()

  if (i + 1) % VALIDATION_INTERVAL == 0:
    validation_accuracy.reset()
    executor.forward(data=validation_X, softmax_label=validation_Y)
    validation_accuracy.update(validation_Y, executor.outputs)
    print 'iteration %d validation accuracy %f' % (i + 1, validation_accuracy.get())

  for index, pair in enumerate(zip(executor.arg_dict, executor.grad_dict)):
    arg, gradient = pair
    optimizer.update(index, arg, gradient)

info_path = '%s-info' % path
history = (
  ta_table,
  va_table,
  loss_table
)
pickle.dump(history, open('info-path'))
