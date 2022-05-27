import os
import cv2
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import crop as ci
import cv2
import numpy as np
import glob
import os

from pprint import pprint as pp

size_max_image = 500
debug_mode = True

print('Loading model.........')
model = load_model('model-auto.h5')
print('Finished loading model..')


def get_result(result):
    if result[0][0] == 1:
        return('A')
    elif result[0][1] == 1:
        return ('B')
    elif result[0][2] == 1:
        return ('C')
    elif result[0][3] == 1:
        return ('D')
    elif result[0][4] == 1:
        return ('E')
    elif result[0][5] == 1:
        return ('F')
    elif result[0][6] == 1:
        return ('G')
    elif result[0][7] == 1:
        return ('H')
    elif result[0][8] == 1:
        return ('I')
    elif result[0][9] == 1:
        return ('J')
    elif result[0][10] == 1:
        return ('K')
    elif result[0][11] == 1:
        return ('L')
    elif result[0][12] == 1:
        return ('M')
    elif result[0][13] == 1:
        return ('N')
    elif result[0][14] == 1:
        return ('O')
    elif result[0][15] == 1:
        return ('P')
    elif result[0][16] == 1:
        return ('Q')
    elif result[0][17] == 1:
        return ('R')
    elif result[0][18] == 1:
        return ('S')
    elif result[0][19] == 1:
        return ('T')
    elif result[0][20] == 1:
        return ('U')
    elif result[0][21] == 1:
        return ('V')
    elif result[0][22] == 1:
        return ('W')
    elif result[0][23] == 1:
        return ('X')
    elif result[0][24] == 1:
        return ('Y')
    elif result[0][25] == 1:
        return ('Z')
    elif result[0][26] == 1:
        return ('a')
    elif result[0][27] == 1:
        return ('b')
    elif result[0][28] == 1:
        return ('c')
    elif result[0][29] == 1:
        return ('d')
    elif result[0][30] == 1:
        return ('e')
    elif result[0][31] == 1:
        return ('f')
    elif result[0][32] == 1:
        return ('g')
    elif result[0][33] == 1:
        return ('h')
    elif result[0][34] == 1:
        return ('i')
    elif result[0][35] == 1:
        return ('j')
    elif result[0][36] == 1:
        return ('k')
    elif result[0][37] == 1:
        return ('l')
    elif result[0][38] == 1:
        return ('m')
    elif result[0][39] == 1:
        return ('n')
    elif result[0][40] == 1:
        return ('o')
    elif result[0][41] == 1:
        return ('p')
    elif result[0][42] == 1:
        return ('q')
    elif result[0][43] == 1:
        return ('r')
    elif result[0][44] == 1:
        return ('s')
    elif result[0][45] == 1:
        return ('t')
    elif result[0][46] == 1:
        return ('u')
    elif result[0][47] == 1:
        return ('v')
    elif result[0][48] == 1:
        return ('w')
    elif result[0][49] == 1:
        return ('x')
    elif result[0][50] == 1:
        return ('y')
    elif result[0][51] == 1:
        return ('z')


def crop_image(path_in):

    image = cv2.imread(path_in)

    if image.shape[1] < 200:
        return image

    # image = cut_of_bottom(image, 1000)

    image = ci.scale_image(image, size_max_image)
    # if (debug_mode):
    #     show_image(image, window_name)

    image = ci.detect_box(image, True)

    # Create out path
    if not os.path.exists(path_in):
        os.makedirs(path_in)

    # Build output file path
    file_name_ext = os.path.basename(path_in)
    file_name, file_extension = os.path.splitext(file_name_ext)
    # file_path = os.path.join(file_iterator, file_name + file_extension)
    file_path = path_in

    # Write out file
    cv2.imwrite(file_path, image)

    # Return cropped image
    return image


filename = 'C:/Users/Timilehin Vincent/Desktop/Desktop/project/CNN_AlphabetRecognition-master/icr_008_t_2.png'

print('------ Loading original image-----')
original_image = image.load_img(
    filename, target_size=(100, 100), color_mode='grayscale')
plt.imshow(original_image)

cropped_image = crop_image(filename)

print('---------------Loading cropped image for prediction-----------')

test_image = image.load_img(
    filename, target_size=(100, 100), color_mode='grayscale')
print('------------Image loaded succesfully------------')
plt.imshow(test_image)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image)
result = get_result(result)
print('\nPredicted Alphabet is: {}'.format(result))
