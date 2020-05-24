import keras

# import Run:AI gradient accumulation
import runai.ga.keras

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = keras.applications.vgg16.preprocess_input(x_train)
x_test = keras.applications.vgg16.preprocess_input(x_test)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

optimizer = keras.optimizers.SGD()

# wrap the Keras.Optimizer with gradient accumulation of 2 steps
optimizer = runai.ga.keras.optimizers.Optimizer(optimizer, steps=2)

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

model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=1,
    verbose=1,
    validation_data=(x_test, y_test),
)
