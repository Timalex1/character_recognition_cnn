import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from sklearn import metrics

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=52, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("I'm here -0")


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

print("I'm here -1")
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory='C:/Users/Timilehin Vincent/Desktop/Desktop/project/work/training',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    directory='C:/Users/Timilehin Vincent/Desktop/Desktop/project/work/test',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

history = model.fit(train_generator,
                    steps_per_epoch=16,
                    epochs=3,
                    validation_data=test_generator,
                    validation_steps=16)

print("I'm here -2")


predicted_classes = np.argmax(history, 1)

# Get ground-truth classes and class-labels
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())


# Use scikit-learn to get statistics
print("/n/n/n/n/n/n/n/n")
print("scikit-learn to get statistics")
report = metrics.classification_report(
    true_classes, predicted_classes, target_names=class_labels)
print(report)


# save model and architecture to single file
model.save("model-auto-1.h5")
