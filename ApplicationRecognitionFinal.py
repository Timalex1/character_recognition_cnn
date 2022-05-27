from keras.layers.convolutional import Cropping2D
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

model = Sequential()
model.add(Cropping2D(cropping=((5, 5), (5, 5)), input_shape=(50, 50, 1)))
model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=52, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# # model.summary()

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory='/home/rsa-key-20210903/images/Training',
    target_size=(300, 300),
    crop_to_aspect_ratio=True,
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    directory='/home/rsa-key-20210903/images/Test',
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

history = model.fit(train_generator,
                    steps_per_epoch=32,
                    epochs=3,
                    validation_data=test_generator,
                    validation_steps=32)

# Using pickle to save dataset model
# pickle.dump(model, open('CNN_model.sav', 'wb'))

# Using Keras to save dataset model
# save model and architecture to single file
# model.save("model.h5")

# model = load_model('model.h5')
# print("Loaded model from disk")


# fnx - to print out result.
def get_result(result):
    if result[0][0] == 1:
        return('a')
    elif result[0][1] == 1:
        return ('b')
    elif result[0][2] == 1:
        return ('c')
    elif result[0][3] == 1:
        return ('d')
    elif result[0][4] == 1:
        return ('e')
    elif result[0][5] == 1:
        return ('f')
    elif result[0][6] == 1:
        return ('g')
    elif result[0][7] == 1:
        return ('h')
    elif result[0][8] == 1:
        return ('i')
    elif result[0][9] == 1:
        return ('j')
    elif result[0][10] == 1:
        return ('k')
    elif result[0][11] == 1:
        return ('l')
    elif result[0][12] == 1:
        return ('m')
    elif result[0][13] == 1:
        return ('n')
    elif result[0][14] == 1:
        return ('o')
    elif result[0][15] == 1:
        return ('p')
    elif result[0][16] == 1:
        return ('q')
    elif result[0][17] == 1:
        return ('r')
    elif result[0][18] == 1:
        return ('s')
    elif result[0][19] == 1:
        return ('t')
    elif result[0][20] == 1:
        return ('u')
    elif result[0][21] == 1:
        return ('v')
    elif result[0][22] == 1:
        return ('w')
    elif result[0][23] == 1:
        return ('x')
    elif result[0][24] == 1:
        return ('y')
    elif result[0][25] == 1:
        return ('z')
    elif result[0][26] == 1:
        return ('A')
    elif result[0][27] == 1:
        return ('B')
    elif result[0][28] == 1:
        return ('C')
    elif result[0][29] == 1:
        return ('D')
    elif result[0][30] == 1:
        return ('E')
    elif result[0][31] == 1:
        return ('F')
    elif result[0][32] == 1:
        return ('G')
    elif result[0][33] == 1:
        return ('H')
    elif result[0][34] == 1:
        return ('I')
    elif result[0][35] == 1:
        return ('J')
    elif result[0][36] == 1:
        return ('K')
    elif result[0][37] == 1:
        return ('L')
    elif result[0][38] == 1:
        return ('M')
    elif result[0][39] == 1:
        return ('N')
    elif result[0][40] == 1:
        return ('O')
    elif result[0][41] == 1:
        return ('P')
    elif result[0][42] == 1:
        return ('Q')
    elif result[0][43] == 1:
        return ('R')
    elif result[0][44] == 1:
        return ('S')
    elif result[0][45] == 1:
        return ('T')
    elif result[0][46] == 1:
        return ('U')
    elif result[0][47] == 1:
        return ('V')
    elif result[0][48] == 1:
        return ('W')
    elif result[0][49] == 1:
        return ('X')
    elif result[0][50] == 1:
        return ('Y')
    elif result[0][51] == 1:
        return ('Z')

# filename = r'C:\\Users\\Timilehin Vincent\\Desktop\\Desktop\\project\\CNN_AlphabetRecognition-master\\Testing\\e\\22.png'
# test_image = image.load_img(filename, target_size=(32, 32))
# plt.imshow(test_image)
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)

# result = model.predict(test_image)
# result = get_result(result)
# print('Predicted Alphabet is: {}'.format(result))
