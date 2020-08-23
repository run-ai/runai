import sys

import keras

# import Run:AI HPO assistance library
import runai.hpo

if len(sys.argv) != 2:
    print("usage: python %s <HPO directory>" % sys.argv[0])
    exit(1)

hpo_dir = sys.argv[1]
print("Using HPO directory %s" % hpo_dir)

# initialize the Run:AI HPO assistance library
runai.hpo.init(hpo_dir)

# pick a configuration for this HPO experiment
# we pass the options of all hyperparameters we want to test
# `config` will hold a single value for each parameter
config = runai.hpo.pick(dict(
    batch_size=[32, 64, 128],
    lr=[1, 0.1, 0.01, 0.001],
))

print('Using configuration: %s' % str(config))

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = keras.applications.vgg16.preprocess_input(x_train)
x_test = keras.applications.vgg16.preprocess_input(x_test)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

optimizer = keras.optimizers.SGD(lr=config['lr'])

model = keras.applications.vgg16.VGG16(
    include_top=True,
    weights=None,
    input_shape=(32,32,3),
    classes=10,
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

class ReportCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        runai.hpo.report(epoch, dict(
            loss=float(logs['loss']),
            acc=float(logs['acc']),
            val_loss=float(logs['val_loss']),
            val_acc=float(logs['val_acc']),
        ))

model.fit(
    x=x_train,
    y=y_train,
    batch_size=config['batch_size'],
    epochs=10,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[ReportCallback()],
)
