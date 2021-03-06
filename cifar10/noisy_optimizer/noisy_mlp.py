import cPickle as pickle
import minpy.nn.model_builder as builder
from facility import *
from noisy_loss import *
from solver_primitives import *
from utilities.data_utility import load_cifar10

from GPU_utility import GPU_availability
from minpy.context import set_context, gpu
set_context(gpu(GPU_availability()[0]))

ACTIVATION = 'ReLU'
SHAPE = (1024,) * 3 + (10,)
BATCH_SIZE = 64

X_SHAPE = (3072,)
activation = getattr(builder, ACTIVATION)
mlp = builder.Sequential()
for shape in SHAPE[:-1]:
  mlp.append(builder.Affine(shape))
  mlp.append(activation())
mlp.append(builder.Affine(SHAPE[-1]))
model = builder.Model(mlp, 'softmax', X_SHAPE)
initialize(model)
updater = Updater(model, 'sgd', {'learning_rate' : 0.01})

training_X, training_Y, validation_X, validation_Y, test_X, test_Y, = \
  load_cifar10(path='../../cifar10/utilities/cifar/', center=True, rescale=True)
X_batches = Batches(training_X, BATCH_SIZE)
Y_batches = Batches(training_Y, BATCH_SIZE)

ITERATIONS = 20000
LOGGING_INTERVAL = 10
VALIDATION_INTERVAL = 50
loss_table = []
validation_accuracy_table = []
sample = lambda N, D : np.random.uniform(-1, 1, (N, D))
for i in range(ITERATIONS):
  X_batch = next(X_batches)
  Y_batch = next(Y_batches)
  gradients, loss = unbiased_noisy_gradient_loss(model, X_batch, Y_batch, sample, 0.5, BATCH_SIZE, 10)
  updater.update(gradients)

  loss = to_float(loss)
  loss_table.append(loss)
  if (i + 1) % LOGGING_INTERVAL == 0:
    print 'iteration %d loss %f' % (i + 1, loss)

  if (i + 1) % VALIDATION_INTERVAL == 0:
    predictions = model.forward(validation_X, 'test')
    validation_accuracy = accuracy(predictions, validation_Y)
    print 'iteration %d validation accuracy %f' % (i + 1, validation_accuracy)
    validation_accuracy_table.append(validation_accuracy)
    
predictions = model.forward(test_X, 'test')
test_accuracy = accuracy(predictions, test_Y)

path = 'noisy-mlp-%s-%s' % (ACTIVATION, '-'.join(str(shape) for shape in SHAPE))
info_path = '%s-info' % path
history = (
  test_accuracy,
  loss_table,
  validation_accuracy_table,
)
pickle.dump(history, open(info_path, 'wb'))
