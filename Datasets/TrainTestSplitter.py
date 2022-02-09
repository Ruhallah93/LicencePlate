import glob
import os
from tqdm import tqdm
import shutil
import random

dataset_path = "/home/ruhiii/Downloads/CharFinder/final_data_set_repaired_sub/"
dst = "/home/ruhiii/Downloads/CharFinder/TrainTest/"
if not os.path.exists(dst + "train"):
    os.makedirs(dst + "train")
if not os.path.exists(dst + "test"):
    os.makedirs(dst + "test")

print("Start Moving ...")
for path in tqdm(glob.glob(dataset_path + "*.txt")):
    title = os.path.basename(path).split("$")[0]
    if len(glob.glob(dataset_path + title + "*.txt")) > 1:
        last = "test"
    else:
        last = "train"
    shutil.copy(path, dst + last)
    src_img = f"{dataset_path + os.path.basename(path).split('.')[0]}.jpg"
    shutil.copy(src_img, dst + last)
