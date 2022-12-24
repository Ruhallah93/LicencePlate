import uuid

import pandas as pd
from threading import Thread
import argparse
import cv2
import random

import pandas
from PIL import Image
import os
from utils.Utils import letter_to_class, visualization, resize, create_noise_dictionary
from utils.Utils import letters
from utils.Utils import crop
from datetime import datetime
from utils.Creator import get_new_plate
import time
import numpy as np
from csv import DictWriter


def adjust_plate_and_mask_size(perspective_plate, mask, img_size):
    b_background = Image.new('RGBA', img_size, (0, 0, 0, 255))
    b_background.paste(Image.fromarray(perspective_plate), (0, 0))
    perspective_plate = np.array(b_background)
    w_background = Image.new('RGBA', img_size, (255, 255, 255, 255))
    w_background.paste(Image.fromarray(mask), (0, 0))
    mask = np.array(w_background)
    return perspective_plate, mask


def save_glyphs_(store_address, plate, perspective_plate, bonding_boxes, glyph_size, name,
                 glyph_state='grayscale', save_mode="alphabet+digit"):
    store_directory = store_address + "glyphs" + os.sep if opt.save_bounding_boxes & opt.save_glyphs else store_address
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
        # w_background = Image.new('RGB', glyph_size, (255, 255, 255))
        # w_background.paste(glyph, (0, 0))
        if glyph_state == 'grayscale':
            w_background = w_background.convert("L")
        _id = uuid.uuid4().__str__()
        cv2.imwrite(store_directory + char + os.sep + name + '$' + _id + ".png", np.array(w_background))


def save_bounding_boxes_(store_directory, plate, perspective_plate, bonding_boxes, name, _id):
    label_file = open("{}.txt".format(store_directory + name + '$' + _id + "txt"), 'w')
    height, width = perspective_plate.height, perspective_plate.width
    classes = [plate[0], plate[1], letter_to_class[plate[2]], plate[3], plate[4], plate[5], plate[6],
               plate[7]]
    for i, box in enumerate(sorted(bonding_boxes, key=lambda x: x[0])):
        x, y, w, h = box
        x_center = int((x + (0.5) * w)) / width
        y_center = int((y + (0.5) * h)) / height
        label_file.write("{} {} {} {} {}\n".format(classes[i], x_center, y_center, w / width, h / height))
    label_file.close()


def generate_and_save_plates(thread_num, store_address, cars, predefined_noises_file, noise_ranges,
                             dataset_size: int = 200,
                             img_size: tuple = (600, 400),
                             save_plate=True,
                             save_bounding_boxes=True,
                             save_mask=True,
                             save_glyphs=True,
                             glyph_size: tuple = (128, 128),
                             mask_state='grayscale',
                             crop_to_content=False,
                             glyph_state='grayscale',
                             save_glyph_mode='alphabet+digit',
                             predefined_noises=False,
                             noise_labeling=False):
    random.seed(datetime.now())

    # rotation_maximums = {'pitch': [30], 'yaw': [30], 'roll': [30], 'pitch+yaw': [30, 30],
    #                      'pitch+yaw+roll': [30, 30, 15]}

    counter = 0
    for i in range(dataset_size):
        plate_size = (600, 132)
        attach_point = (int((img_size[1] - plate_size[1]) / 2), int((img_size[0] - plate_size[0]) / 2))

        # Select randomly a car
        postfix_path = random.choice(
            [x for x in os.listdir(cars) if os.path.isfile(os.path.join(cars, x))])
        car_path = cars + os.sep + postfix_path

        if predefined_noises:
            noise_vectors = pandas.read_csv(predefined_noises_file)
            noise_dic = noise_vectors.iloc[thread_num * dataset_size + i].to_dict()
        else:
            # random noises
            noise_dic = create_noise_dictionary(noise_ranges)

        plate, perspective_plate, mask, bonding_boxes, plate_box, noises = get_new_plate(img_size, noise_dic,
                                                                                         plate_size=plate_size,
                                                                                         mask_state=mask_state,
                                                                                         paste_point=attach_point,
                                                                                         background_path=car_path)
        # Keep only plate
        if crop_to_content:
            perspective_plate, mask, plate_box, bonding_boxes = crop(perspective_plate, mask, plate_box, bonding_boxes)

        # Adjust size
        perspective_plate, mask = adjust_plate_and_mask_size(perspective_plate, mask, img_size)

        black_white_random = random.randint(1, 10)
        # Numpy array to PIL
        if black_white_random == 1:
            perspective_plate = cv2.cvtColor(cv2.cvtColor(perspective_plate, cv2.COLOR_RGBA2GRAY), cv2.COLOR_GRAY2BGR)
        else:
            perspective_plate = cv2.cvtColor(perspective_plate, cv2.COLOR_RGBA2BGR)

        perspective_plate = Image.fromarray(perspective_plate)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2BGR)
        mask = Image.fromarray(mask)

        if save_bounding_boxes:
            if len(bonding_boxes) != 8:
                counter += 1
                print("len(merged_boxes): ", len(bonding_boxes))
                visualization(perspective_plate, [mask], boxes=bonding_boxes + [plate_box], waitKey=0)
                continue

        # Visualization
        # t : 116
        if noise_labeling:
            # print(noises['pred_label'])
            k = visualization(perspective_plate, [mask], waitKey=0)
        else:
            k = -1

        _id = uuid.uuid4().__str__()
        name = plate[0] + plate[1] + '_' + plate[2] + '_' + plate[3] + plate[4] + plate[5] + plate[6] + plate[7]

        base_address = store_address + os.sep if store_address[-1] != os.sep else store_address
        if not os.path.exists(base_address):
            os.makedirs(base_address)

        # Save noise
        if noise_labeling:
            if k == 116:
                noises['label'] = 'Correct'
            else:
                noises['label'] = 'Corrupt'
        else:
            noises['label'] = 'Unknown'
        noises['instance_name'] = name + '$' + _id
        with open(base_address + 'noise_vectors.csv', 'a') as f:
            w = DictWriter(f, fieldnames=list(noises.keys()))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(noises)
            f.close()

        # Update Store Directory
        if noise_labeling:
            base_address = base_address + noises['label'] + os.sep

        store_directory = base_address + "images" + os.sep if save_mask else base_address
        if not os.path.exists(store_directory):
            os.makedirs(store_directory)

        # Save bounding boxes
        if save_bounding_boxes:
            save_bounding_boxes_(store_directory, plate, perspective_plate, bonding_boxes, name, _id)

        # Save Glyphs
        if save_glyphs:
            save_glyphs_(base_address, plate, perspective_plate, bonding_boxes, glyph_size,
                         glyph_state=glyph_state,
                         save_mode=save_glyph_mode,
                         name=name)

        # Save plates
        if save_plate:
            cv2.imwrite(store_directory + name + '$' + _id + ".png", np.array(perspective_plate))

        # Save masks
        if save_mask:
            store_directory = base_address + "masks" + os.sep
            if not os.path.exists(store_directory):
                os.makedirs(store_directory)
            cv2.imwrite(store_directory + name + '$' + _id + ".png", np.array(mask))

    print("fails: ", counter)


if __name__ == '__main__':
    # For test: set workers default to 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=80000, help='number of plates to generate')
    parser.add_argument('--workers', type=int, default=10, help='number of threads to run')
    parser.add_argument('--img_size', nargs='+', type=int, default=[1000, 800], help='size of background')
    parser.add_argument('--save_plate', action='store_true', help='save the masks if true')
    parser.add_argument('--save_bounding_boxes', action='store_true', help='save the bounding boxes if true')
    parser.add_argument('--save_mask', action='store_true', help='save the masks if true')
    parser.add_argument('--save_glyphs', action='store_true', help='save the masks if true')
    parser.add_argument('--crop_to_content', action='store_true', help='save only plate')
    parser.add_argument('--glyph_size', nargs='+', type=int, default=[32, 32], help='size of saved glyphs')
    parser.add_argument('--glyph_state', type=str, default='colorful', help='grayscale or colorful')
    parser.add_argument('--save_glyph_mode', type=str, default='alphabet+digit', help='alphabet+digit|alphabet|digit')
    parser.add_argument('--mask_state', type=str, default='grayscale', help='grayscale or colorful')
    parser.add_argument('--address', type=str, default='output/CapsNet_data/', help='The address of saving dataset')
    parser.add_argument('--cars', type=str, default='files/cars')
    parser.add_argument('--noise_labeling', action='store_true', help='save the masks if true')
    parser.add_argument('--predefined_noises', action='store_true', help='generate random noises. false: read from csv')
    parser.add_argument('--predefined_noises_file', type=str, default='utils/noise/noise_vectors.csv')
    parser.add_argument('--noise_ranges', type=str, default='utils/noise/noises_parameters_ranges.csv')
    opt = parser.parse_args()

    opt.save_plate = False
    opt.save_mask = False
    opt.save_bounding_boxes = False
    opt.save_glyphs = True
    opt.crop_to_content = True
    opt.noise_labeling = False
    opt.predefined_noises = True
    opt.mask_state = "grayscale"

    address = opt.address + os.sep if opt.address[-1] != os.sep else opt.address

    size = opt.size
    max_threads = opt.workers
    threadList = []
    if max_threads == 1:
        generate_and_save_plates(0, address, opt.cars, opt.predefined_noises_file, opt.noise_ranges,
                                 size, tuple(opt.img_size),
                                 opt.save_plate, opt.save_bounding_boxes,
                                 opt.save_mask, opt.save_glyphs, tuple(opt.glyph_size),
                                 opt.mask_state, opt.crop_to_content,
                                 opt.glyph_state, opt.save_glyph_mode,
                                 opt.predefined_noises, opt.noise_labeling)
    else:
        for i in range(max_threads):
            print("Tread ", i + 1, " is running")
            chunk_size = (size // max_threads) if i < max_threads - 1 else (size // max_threads) + (size % max_threads)
            t = Thread(target=generate_and_save_plates,
                       args=(i, address, opt.cars, opt.predefined_noises_file, opt.noise_ranges,
                             chunk_size, tuple(opt.img_size),
                             opt.save_plate, opt.save_bounding_boxes,
                             opt.save_mask, opt.save_glyphs, tuple(opt.glyph_size),
                             opt.mask_state, opt.crop_to_content,
                             opt.glyph_state, opt.save_glyph_mode,
                             opt.predefined_noises, opt.noise_labeling))
            t.start()
            threadList.append(t)
            if i == 0:
                time.sleep(4)
        for t in threadList:
            t.join()
