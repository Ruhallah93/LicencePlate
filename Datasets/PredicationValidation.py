import glob
import os
import numpy as np
import shutil
from PIL import Image
import cv2

dataset_path = "/home/ruhiii/Downloads/CharFinder/Predication/wrong_detected/"

weakness = "/home/ruhiii/Downloads/CharFinder/Predication/Weakness/"
unstructured = "/home/ruhiii/Downloads/CharFinder/Predication/Unstructured/"
ambiguous = "/home/ruhiii/Downloads/CharFinder/Predication/Ambiguous/"

if not os.path.exists(weakness):
    os.makedirs(weakness)
if not os.path.exists(unstructured):
    os.makedirs(unstructured)
if not os.path.exists(ambiguous):
    os.makedirs(ambiguous)

list = glob.glob(dataset_path + "*.jpg")

i = 0
while i < len(list):
    title = os.path.basename(list[i]).split('.')[0]
    print(i, "/", len(list), "\t", title)
    img = Image.open(list[i])
    img = np.array(img)
    cv2.putText(img, os.path.basename(title).split('.')[0], (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1,
                cv2.LINE_AA)
    cv2.imshow('img', img)
    k = cv2.waitKey(0)
    # w : 119
    # u : 117
    # a : 97
    # b : 98
    if k == 98:
        print("go back")
        i -= 1
    elif k == 119:
        shutil.copy(list[i], weakness)
        print(title, " -> Weakness.")
        i += 1
    elif k == 117:
        shutil.copy(list[i], unstructured)
        print(title, " -> Unstructured.")
        i += 1
    elif k == 97:
        shutil.copy(list[i], ambiguous)
        print(title, " -> Ambiguous.")
        i += 1
