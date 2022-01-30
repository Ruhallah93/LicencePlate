import glob
import os
import shutil
from tqdm import tqdm

dataset_path = "/home/ruhiii/Downloads/CharFinder/data_temp_repaired/"
pre_dst = "/home/ruhiii/Downloads/CharFinder/Tasks/"

counter = 0
for src in tqdm(glob.glob(dataset_path + "*.txt")):
    src_img = f"{dataset_path + os.path.basename(src).split('.')[0]}.jpg"
    if counter < 1500:
        dst = pre_dst + "Bideh/"
        shutil.copy(src, dst)
        shutil.copy(src_img, dst)
    elif 1500 <= counter < 3000:
        dst = pre_dst + "Amin/"
        shutil.copy(src, dst)
        shutil.copy(src_img, dst)
    elif 3000 <= counter < 4000:
        dst = pre_dst + "Marjane/"
        shutil.copy(src, dst)
        shutil.copy(src_img, dst)
    elif counter >= 4000:
        dst = pre_dst + "Rouhollah/"
        shutil.copy(src, dst)
        shutil.copy(src_img, dst)

    counter += 1
