import glob
import os
from tqdm import tqdm
import cv2
import json

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


def save_yolo_format(image_file, json_file, save_label_path, save_image_path):
    if not os.path.exists(save_label_path):
        os.makedirs(save_label_path)
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)

    with open(json_file) as f:
        chars = json.loads(f.read())

    chars = sorted(chars, key=lambda x: x['x'])

    bgr = cv2.imread(image_file)

    result = ""
    for char in chars:
        x_center = (char["x"] + char["width"] // 2) / bgr.shape[1]
        y_center = (char["y"] + char["height"] // 2) / bgr.shape[0]

        width = char["width"] / bgr.shape[1]
        height = char["height"] / bgr.shape[0]

        label = origin_id_to_ours[char["char_en"]]

        result += str(label) + " " + str(x_center) + " " + str(y_center) + " " + str(width) + " " + str(height) + "\n"

    if (len(chars) == 8):
        name_of_file = "{}_{}_{}".format(chars[0]["char_en"] + chars[1]["char_en"],
                                         origin_label_to_ours[chars[2]["char_en"]],
                                         chars[3]["char_en"] + chars[4]["char_en"] + chars[5]["char_en"] + chars[6][
                                             "char_en"] + chars[7]["char_en"])
    else:
        name_of_file = "faulty"

    counter = 1
    while (os.path.exists(save_label_path + name_of_file + '$' + str(counter) + '.txt')):
        counter += 1
    with open(save_label_path + name_of_file + '$' + str(counter) + '.txt', "w+") as f:
        f.write(result)

    cv2.imwrite(save_image_path + name_of_file + '$' + str(counter) + '.jpg', bgr)


print("Start converting json to text files ...")
for path in tqdm(glob.glob("/home/ruhiii/Downloads/CharFinder/labels/*.json")):
    save_yolo_format(f"/home/ruhiii/Downloads/CharFinder/images/{os.path.basename(path).split('.')[0]}.jpg",
                     path,
                     "/home/ruhiii/Downloads/CharFinder/data_temp/",
                     "/home/ruhiii/Downloads/CharFinder/data_temp/")
