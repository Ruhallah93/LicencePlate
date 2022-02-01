import glob
import os
import numpy as np
import shutil
from PIL import Image
import cv2

dataset_path = "/home/ruhiii/Downloads/CharFinder/Tasks/Rouhollah/"
dst = "/home/ruhiii/Downloads/CharFinder/Tasks/Rouhollah_Incorrect/"
ambihuous = "/home/ruhiii/Downloads/CharFinder/Tasks/Rouhollah_Ambihuous/"
if not os.path.exists(dst):
    os.makedirs(dst)
if not os.path.exists(ambihuous):
    os.makedirs(ambihuous)

list = glob.glob(dataset_path + "*.txt")

i = 0
while i < len(list):
    print(i, "/", len(list), "\t", os.path.basename(list[i]).split('.')[0])
    src_img = f"{dataset_path + os.path.basename(list[i]).split('.')[0]}.jpg"
    img = Image.open(src_img)
    img = np.array(img)
    cv2.putText(img, os.path.basename(list[i]).split('.')[0], (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1,
                cv2.LINE_AA)
    cv2.imshow('img', img)
    k = cv2.waitKey(0)
    # b : 98
    # o : 111
    # t : 116
    if k == 98:
        print("go back")
        i -= 1
    elif k == 111:
        shutil.copy(list[i], dst)
        shutil.copy(src_img, dst)
        print(os.path.basename(list[i]).split('.')[0], " -> Incorrects.")
        i += 1
    elif k == 116:
        shutil.copy(list[i], ambihuous)
        shutil.copy(src_img, ambihuous)
        print(os.path.basename(list[i]).split('.')[0], " -> Ambihuous.")
        i += 1
    else:
        i += 1
