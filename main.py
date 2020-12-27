import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# Define the directories
train_dir = 'D:\\Datasets\\Mini Project Dataset 1\\train'
test_dir = 'D:\\Datasets\\Mini Project Dataset 1\\test'

# Data Augmenting
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   vertical_flip=True,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  vertical_flip=True,
                                  horizontal_flip=True
                                  )

# Loading and cropping Images
training_generator = train_datagen.flow_from_directory(
    train_dir,
    class_mode='categorical',
    classes=['gesture1', 'gesture2', 'gesture3', 'gesture4'],
    target_size=(200, 200),
    batch_size=100
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    class_mode='categorical',
    classes=['gesture1', 'gesture2', 'gesture3', 'gesture4'],
    target_size=(200, 200),
    batch_size=10
)

# Model Definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(64, (2, 2), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax'),
])

# Compiling the Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Fitting the data and starting the training
history = model.fit(training_generator, epochs=10, validation_data=validation_generator, verbose=1)


# Saving Model as Keras model
model.save('Gesture_recogniser.h5')

# Converting and optimising model to tflite format
model = tf.keras.models.load_model("./Gesture_recogniser.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
open('Gesture_recogniser_quantised.tflite', 'wb').write(tflite_model)


# Plotting the Results...
acc = history.history['accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')
plt.show()

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Validation accuracy')
plt.show()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.show()

plt.plot(epochs, val_loss, 'b', label='Training Loss')
plt.title('Validation loss')
plt.show()

