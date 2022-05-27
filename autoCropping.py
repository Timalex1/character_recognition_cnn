import glob
import os
import cv2
import numpy as np

files = glob.glob(
    "C:/Users/Timilehin Vincent/Desktop/Desktop/project/imageCopy/Training/X/**/*.png", recursive=True)

for f in files:
    try:
        img_name = f
        print(img_name)
        img = cv2.imread(img_name)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours and hierarchy, use RETR_TREE for creating a tree of contours within contours
        # [-2:] indexing takes return value before last (due to OpenCV compatibility issues).
        cnts, hiers = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

        # https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
        # Hierarchy Representation in OpenCV
        # So each contour has its own information regarding what hierarchy it is, who is its child, who is its parent etc.
        # OpenCV represents it as an array of four values : [Next, Previous, First_Child, Parent]
        parent = hiers[0, :, 3]

        # Find parent contour with the maximum number of child contours
        # Use np.bincount for counting the number of instances of each parent value.
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html#numpy.bincount
        hist = np.bincount(np.maximum(parent, 0))
        max_n_childs_idx = hist.argmax()

        # Get the contour with the maximum child contours
        c = cnts[max_n_childs_idx]

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(c)

        # Crop the bounding rectangle out of img
        img = img[y:y+h, x:x+w, :]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Convert to binary image (after cropping) and invert polarity
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        #cv2.imshow('thresh', thresh);cv2.waitKey(0);cv2.destroyAllWindows()

        # Find connected components (clusters)
        nlabel, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thresh, connectivity=8)

        # Delete large, small, tall and wide clusters - not letters for sure
        max_area = 2000
        min_area = 10
        max_width = 100
        max_height = 100
        for i in range(1, nlabel):
            if (stats[i, cv2.CC_STAT_AREA] > max_area) or \
                (stats[i, cv2.CC_STAT_AREA] < min_area) or \
                (stats[i, cv2.CC_STAT_WIDTH] > max_width) or \
                    (stats[i, cv2.CC_STAT_HEIGHT] > max_height):
                thresh[labels == i] = 0

        #cv2.imshow('thresh', thresh);cv2.waitKey(0);cv2.destroyAllWindows()

        # Use "closing" morphological operation for uniting text area
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((50, 50)))

        # Find contours once more
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_NONE)[-2]

        # Get contour with maximum area
        c = max(cnts, key=cv2.contourArea)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(c)

        # Crop the bounding rectangle out of img (leave some margins)
        out = img[y-5:y+h+5, x-5:x+w+5]

        # Show output
        # cv2.imshow('out', out)

        cv2.imwrite(img_name, out)
        # cv2.imwrite()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))
