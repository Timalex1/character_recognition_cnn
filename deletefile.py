import os
import glob

files = glob.glob(
    "C:/Users/Timilehin Vincent/Desktop/Desktop/project/SVM ONLINE RECOGNITION DATASET/training/**/*.db", recursive=True)

for f in files:
    try:
        os.remove(f)
        print(1)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))
