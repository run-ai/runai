import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

import runai.reporter
import runai.reporter.keras

NUM_CLASSES = 10
BATCH_SIZE = 128

runai.reporter.keras.autolog()
runai.reporter.reportParameter("current_state", "preprocessing")

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

runai.reporter.reportParameter("current_state", "final tunning")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.summary()

optimizer = keras.optimizers.Adam()

runai.reporter.reportParameter("current_state", "compiling")

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer)

runai.reporter.reportParameter("current_state", "fitting")
model.fit(x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=10,
    verbose=1,
    validation_data=(x_test, y_test))

runai.reporter.reportParameter("current_state", "done")

# gracefully shut down the Run:AI reporting library
runai.reporter.finish()
