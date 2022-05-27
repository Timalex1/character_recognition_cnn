from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pickle
from skimage.io import imread
from skimage.transform import resize


def load_image_files(container_path, dimension=(64, 75, 1)):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to

    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir()
               if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            print(file.name)
            img = imread(file)
            img_resized = resize(
                img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)


image_dataset = load_image_files("/home/rsa-key-20210903/images/Training")

print("Loaded successfully")


X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3, random_state=109)

pca = PCA(n_components=0.99)  # adjust yourself
pca.fit(X_train)

X_t_train = pca.transform(X_train)
X_t_test = pca.transform(X_test)

print("----done---")

param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

svc = svm.SVC()

# clf = LogisticRegression()  # GridSearchCV(svc, param_grid)
clf = GridSearchCV(svc, param_grid)

clf.fit(X_t_train, y_train)

y_pred = clf.predict(X_t_test)

print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))

file = open("/home/timilehinvincent/text1.txt", "w")
print("-------------Creating text file-----")
file.write("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))
print("-------------Successfully written to text file-----")


print("-------------Creating SVM file-----")
# Save SVM model in pickle file
pickle.dump(clf, open("/home/timilehinvincent/Classification_Model1.p", "wb"))
print("-------------Successfully Created model file-----")
