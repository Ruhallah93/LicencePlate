import glob
import os
from tqdm import tqdm
import cv2

origin_id_to_ours = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'alef': 10,
                     'b': 11, 'p': 12, 't': 13, 'the': 14, 'j': 15, 'd': 19, 'ze': 22, 's': 24, 'she': 25, 'sad': 26,
                     'ta': 28, 'ein': 30, 'fe': 32, 'q': 33, 'kaf': 34, 'gaf': 35, 'l': 36, 'm': 37, 'n': 38, 'v': 39,
                     'h': 40, 'y': 41, 'malol': 42, 'diplomat': 43, 'siyasi': 44, 'tashrifat': 45}

origin_label_to_ours = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8',
                        '9': '9', 'alef': 'ALEF', 'b': 'BE', 'j': 'JIM', 'l': 'LAM', 'm': 'MIM', 'n': 'NON',
                        'q': 'GHAF', 'v': 'VAV', 'h': 'HE', 'y': 'YE', 'd': 'DAL', 's': 'SIN', 'sad': 'SAD',
                        'malol': 'WHEEL', 't': 'TE', 'ta': 'TA', 'ein': 'EIN', 'diplomat': "DPLMT", 'siyasi': "SYSI",
                        'p': 'PE', 'the': 'SE', 'ze': 'ZE', 'she': 'SHIN', 'fe': 'FE', 'kaf': 'KAF',
                        'tashrifat': 'TSHFT', 'gaf': 'GAF'}

dataset_path = "/home/ruhiii/Downloads/CharFinder/final_data_set/"
dst = "/home/ruhiii/Downloads/CharFinder/final_data_set_repaired/"


def repair(from_path, to_path):
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    image_file = f"{dataset_path + os.path.basename(from_path).split('.')[0]}.jpg"
    bgr = cv2.imread(image_file)

    with open(from_path, "r") as f:
        result = f.read()
        lines = result.split('\n')
        data = [line.split(None, 1)[0] for line in lines[:-1]]

    if len(data) == 8:
        key = [name for name, value in origin_id_to_ours.items() if str(value) == data[2]][0]
        data[2] = origin_label_to_ours[key]
        name_of_file = "{}_{}_{}".format(data[0] + data[1], data[2], data[3] + data[4] + data[5] + data[6] + data[7])
    elif len(data) == 6 and data[2] == '35':
        key = [name for name, value in origin_id_to_ours.items() if str(value) == data[2]][0]
        data[2] = origin_label_to_ours[key]
        name_of_file = "{}_{}_{}".format(data[0] + data[1], data[2], data[3] + data[4] + data[5])
    else:
        name_of_file = "faulty"

    counter = 1
    while (os.path.exists(to_path + name_of_file + '$' + str(counter) + '.txt')):
        counter += 1
    with open(to_path + name_of_file + '$' + str(counter) + '.txt', "w+") as f:
        f.write(result)

    cv2.imwrite(to_path + name_of_file + '$' + str(counter) + '.jpg', bgr)


print("Start Repairing ...")
for path in tqdm(glob.glob(dataset_path + "*.txt")):
    repair(path, dst)
