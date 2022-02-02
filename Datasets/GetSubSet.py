import glob
import os
from tqdm import tqdm
import shutil

support_letter = ["BE", "DAL", "EIN", "GHAF", "HE", "JIM", "LAM", "MIM",
                  "NON", "RE", "SAD", "SIN", "TA", "TE", "VAV", "WHEEL", "YE"]

dataset_path = "/home/ruhiii/Downloads/CharFinder/final_data_set_repaired/"
dst = "/home/ruhiii/Downloads/CharFinder/final_data_set_repaired_sub/"
if not os.path.exists(dst):
    os.makedirs(dst)

print("Start Moving ...")
for path in tqdm(glob.glob(dataset_path + "*.txt")):
    if os.path.basename(path).__contains__("faulty"):
        continue
    if support_letter.__contains__(os.path.basename(path).split("_")[1]):
        shutil.copy(path, dst)
        src_img = f"{dataset_path + os.path.basename(path).split('.')[0]}.jpg"
        shutil.copy(src_img, dst)
