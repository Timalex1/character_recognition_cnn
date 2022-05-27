import joblib
from keras.preprocessing import image
from skimage.io import imread
from skimage.transform import resize
from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


def get_result(result):
    if result == [0]:
        return('A')
    elif result == [1]:
        return ('B')
    elif result == [2]:
        return ('C')
    elif result == [3]:
        return ('D')
    elif result == [4]:
        return ('E')
    elif result == [5]:
        return ('F')
    elif result == [6]:
        return ('G')
    elif result == [7]:
        return ('H')
    elif result == [8]:
        return ('I')
    elif result == [9]:
        return ('J')
    elif result == [10]:
        return ('K')
    elif result == [11]:
        return ('L')
    elif result == [12]:
        return ('M')
    elif result == [13]:
        return ('N')
    elif result == [14]:
        return ('O')
    elif result == [15]:
        return ('P')
    elif result == [16]:
        return ('Q')
    elif result == [17]:
        return ('R')
    elif result == [18]:
        return ('S')
    elif result == [19]:
        return ('T')
    elif result == [20]:
        return ('U')
    elif result == [21]:
        return ('V')
    elif result == [22]:
        return ('W')
    elif result == [23]:
        return ('X')
    elif result == [24]:
        return ('Y')
    elif result == [25]:
        return ('Z')
    elif result == [26]:
        return ('a')
    elif result == [27]:
        return ('b')
    elif result == [28]:
        return ('c')
    elif result == [29]:
        return ('d')
    elif result == [30]:
        return ('e')
    elif result == [31]:
        return ('f')
    elif result == [32]:
        return ('g')
    elif result == [33]:
        return ('h')
    elif result == [34]:
        return ('i')
    elif result == [35]:
        return ('j')
    elif result == [36]:
        return ('k')
    elif result == [37]:
        return ('l')
    elif result == [38]:
        return ('m')
    elif result == [39]:
        return ('n')
    elif result == [40]:
        return ('o')
    elif result == [41]:
        return ('p')
    elif result == [42]:
        return ('q')
    elif result == [43]:
        return ('r')
    elif result == [44]:
        return ('s')
    elif result == [45]:
        return ('t')
    elif result == [46]:
        return ('u')
    elif result == [47]:
        return ('v')
    elif result == [48]:
        return ('w')
    elif result == [49]:
        return ('x')
    elif result == [50]:
        return ('y')
    elif result == [51]:
        return ('z')


dimension = (64, 75, 1)

model = joblib.load(open("train_rbf_SVM.pkl", "rb"))

pca = joblib.load(open("train_pca.pkl", "rb"))

print('successfully loaded')

filename = 'r.png'
flat_data = []

img = imread(filename)

img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')

flat_data.append(img_resized.flatten())

flat_data = np.array(flat_data)

test_img = pca.transform(flat_data)

print('Image loaded------------')
plt.imshow(img_resized)

result = model.predict(test_img)

result = get_result(result)

print('Predicted Alphabet is: {}'.format(result))
