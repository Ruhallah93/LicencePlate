import glob
import os
import uuid
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import PIL.ImageOps

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
letters = ["BE", "TE", "JIM", "DAL", "RE", "SIN", "SAD", "TA", "EIN", "GHAF", "LAM", "MIM", "NON", "VAV", "HE",
           "YE", "WHEEL"]


def rgba_2_bgr(img):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    if img.shape[2] == 4:
        street = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return np.array(Image.fromarray(street))
    else:
        return np.array(img)


def visualization(main_image, images=None, boxes=None, waitKey=0):
    main_image = rgba_2_bgr(main_image)
    if boxes is not None:
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(main_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imshow('main', main_image)
    if images is not None:
        for i, img in enumerate(images):
            cv2.imshow(str(i), rgba_2_bgr(img))
    cv2.waitKey(waitKey)


def resize(img, sizes):
    if img.size[0] > sizes[0]:
        img = img.resize((sizes[0], int(sizes[0] * img.size[1] / img.size[0])))
    if img.size[1] > sizes[1]:
        img = img.resize((int(sizes[1] * img.size[1] / img.size[1]), sizes[0]))
    w_background = Image.new('RGB', sizes, (255, 255, 255))
    w_background.paste(img, (0, 0))
    return w_background


def save_glyphs_(store_address, plate, perspective_plate, bonding_boxes, glyph_size,
                 glyph_state='grayscale', save_mode="alphabet+digit"):
    store_directory = store_address + "glyphs" + os.sep
    if not os.path.exists(store_directory):
        os.makedirs(store_directory)
    if save_mode == "alphabet":
        directories = letters
    elif save_mode == "digit":
        directories = [str(i) for i in range(10)]
    else:
        directories = [str(i) for i in range(10)] + letters
    for char in directories:
        if not os.path.exists(store_directory + char + os.sep):
            os.makedirs(store_directory + char + os.sep)

    # bonding_boxes.sort()
    for i, (box, char) in enumerate(zip(bonding_boxes, plate)):
        if save_mode == "alphabet" and i != 2:
            continue
        elif save_mode == "digit" and i == 2:
            continue
        x, y, w, h = box
        glyph = perspective_plate.crop((x, y, x + w, y + h))
        w_background = resize(glyph, glyph_size)
        if glyph_state == 'grayscale':
            w_background = w_background.convert("L")
        _id = uuid.uuid4().__str__()
        cv2.imwrite(store_directory + char + os.sep + _id + ".png", np.array(w_background))


dataset_path = "/home/ruhiii/Downloads/CharFinder/final_data_set_repaired_sub/"
save_path = "/home/ruhiii/Downloads/CharFinder/glyphs_sub_digit/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
for path in tqdm(glob.glob(dataset_path + "*.txt")):
    title = os.path.basename(path).split('.')[0]
    src_img = f"{dataset_path + title}.jpg"
    img = Image.open(src_img).convert("RGBA")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
    img = Image.fromarray(img)
    shape = np.shape(img)
    with open(dataset_path + title + ".txt", "r") as f:
        result = f.read()
        lines = result.split('\n')
        plate = [line.split(None, 1)[0] for line in lines[:-1]]
        key = [name for name, value in origin_id_to_ours.items() if str(value) == plate[2]][0]
        plate[2] = origin_label_to_ours[key]

        boxes = np.array([line.split(" ")[1:] for line in lines[:-1]])
        boxes = boxes.astype(np.float)
        boxes[:, 2] *= shape[1]
        boxes[:, 3] *= shape[0]
        boxes[:, 0] *= shape[1]
        boxes[:, 1] *= shape[0]
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes = boxes.astype(np.int).tolist()

    # visualization(img, boxes=boxes)
    # save_glyphs_(store_address=save_path,
    #              plate=plate,
    #              perspective_plate=img,
    #              bonding_boxes=boxes,
    #              glyph_size=(32, 32),
    #              glyph_state='colorful',
    #              save_mode="digit")
