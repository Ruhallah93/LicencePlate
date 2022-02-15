from threading import Thread
import argparse
import cv2
import uuid
import random
from PIL import Image
import os
from datetime import datetime
from utils.Creator import get_new_plate
from utils.Utils import letter_to_class, visualization
from utils.Utils import rgba_2_bgr
import time
import numpy as np


def save_whole_plate_boxes_(store_address, street, whole_plate_boxes, _id):
    label_file = open("{}.txt".format(store_address + _id + "txt"), 'w')
    height, width = street.shape[0], street.shape[1]
    classes = [0]
    for box in sorted(whole_plate_boxes, key=lambda x: x[0]):
        x, y, w, h = box
        x_center = int((x + (0.5) * w)) / width
        y_center = int((y + (0.5) * h)) / height
        label_file.write("{} {} {} {} {}\n".format(classes[0], x_center, y_center, w / width, h / height))
    label_file.close()


def save_plate_bounding_boxes_(store_directory, plate, perspective_plate, bonding_boxes, name):
    if not os.path.exists(store_directory):
        os.makedirs(store_directory)
    label_file = open("{}.txt".format(store_directory + name), 'w')
    height, width = perspective_plate.shape[0], perspective_plate.shape[1]
    classes = [plate[0], plate[1], letter_to_class[plate[2]], plate[3], plate[4], plate[5], plate[6], plate[7]]
    for i, box in enumerate(sorted(bonding_boxes, key=lambda x: x[0])):
        x, y, w, h = box
        x_center = int((x + (0.5) * w)) / width
        y_center = int((y + (0.5) * h)) / height
        label_file.write("{} {} {} {} {}\n".format(classes[i], x_center, y_center, w / width, h / height))
    label_file.close()


def generate_and_save_streets(store_address, backgrounds, cars,
                              dataset_size: int = 200, img_size: tuple = (600, 400),
                              grid_size: tuple = (3, 3)):
    random.seed(datetime.now())

    store_address = store_address + os.sep if store_address[-1] != os.sep else store_address
    if not os.path.exists(store_address):
        os.makedirs(store_address)

    rotation_maximums = {'pitch': [20], 'yaw': [20], 'roll': [15], 'pitch+yaw': [15, 15],
                         'pitch+yaw+roll': [15, 10, 10]}

    for i in range(dataset_size):
        street_id = uuid.uuid4().__str__()
        # Select randomly a background
        postfix_path = random.choice(
            [x for x in os.listdir(backgrounds) if os.path.isfile(os.path.join(backgrounds, x))])
        background_path = backgrounds + os.sep + postfix_path
        street = np.array(Image.open(background_path).convert("RGBA").resize(img_size))

        whole_plate_boxes = []
        y = 0
        for row in range(grid_size[0], 0, -1):
            # Number of column in this row
            n_column = grid_size[1] + row - 1
            # The dimension of each cell in this row
            w = int(img_size[0] / n_column)
            h = int(img_size[1] / grid_size[0])
            # The size of plate
            plate_size = (600, 132)
            p_w = int(w * 0.67)
            plate_size = (p_w, int(plate_size[1] * p_w / plate_size[0]))
            # The padding size
            # pad_i = int(p_w * 50 / 312)
            pad_i = 0
            pad = (pad_i, pad_i, pad_i, pad_i)
            # The plate+car size
            frame_size = (plate_size[0] + pad_i + 50, plate_size[1] + pad_i + 50)
            # Past plate to middle of car
            paste_point = (int((frame_size[0] - plate_size[0]) / 2), int((frame_size[1] - plate_size[1]) / 2))

            x = 0
            for column in range(n_column):
                # Ignore this cell with probability of 20%
                if random.randrange(10) >= 8:
                    x += w
                    continue
                # Select randomly a car
                postfix_path = random.choice(
                    [x for x in os.listdir(cars) if os.path.isfile(os.path.join(cars, x))])
                car_path = cars + os.sep + postfix_path
                # Create a plate
                plate, perspective_plate, mask, bonding_boxes, plate_box = get_new_plate(
                    img_size=frame_size,
                    rotation_maximums=rotation_maximums,
                    plate_size=plate_size,
                    paste_point=paste_point,
                    mask_state="grayscale",
                    pad=pad,
                    random_scale=0,
                    background_path=car_path)
                # Calculating the plate size regarding padding and car
                nw = frame_size[0] + 2 * pad[0]
                nh = frame_size[1] + 2 * pad[0]
                ny = y + random.randrange(h - nh)
                nx = x + random.randrange(w - nw)
                # Paste plate to street
                street = Image.fromarray(street)
                street.paste(Image.fromarray(perspective_plate), (nx, ny), Image.fromarray(perspective_plate))
                street = np.array(street)
                # Add plate box
                position = (plate_box[0] + nx, plate_box[1] + ny, plate_box[2], plate_box[3])
                whole_plate_boxes.append(position)
                # Save plate boxes
                save_plate_bounding_boxes_(store_address + os.sep + street_id + os.sep,
                                           plate, perspective_plate, bonding_boxes, str(position))

                # Goto next cell
                x += w
            # Goto next row
            y += h

        # Visualization
        # visualization(street, boxes=whole_plate_boxes, waitKey=0)

        # RGBA 2 BGR
        street = rgba_2_bgr(street)

        # Save street
        cv2.imwrite(store_address + street_id + ".png", street)

        # Save bounding boxes
        save_whole_plate_boxes_(store_address, street, whole_plate_boxes, street_id)


if __name__ == '__main__':
    # For test: set workers default to 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1000, help='number of plates to generate')
    parser.add_argument('--workers', type=int, default=10, help='number of threads to run')
    parser.add_argument('--img_size', nargs='+', type=int, default=[1000, 600], help='size of background')
    parser.add_argument('--grid_size', nargs='+', type=int, default=[3, 3], help='# rows, # cell in last row')
    parser.add_argument('--address', type=str, default='output/streets', help='The address of saving dataset')
    parser.add_argument('--backgrounds', type=str, default='files/streets')
    parser.add_argument('--cars', type=str, default='files/cars')
    opt = parser.parse_args()

    address = opt.address + os.sep if opt.address[-1] != os.sep else opt.address
    if not os.path.exists(address):
        os.makedirs(address)

    size = opt.size
    max_threads = opt.workers
    threadList = []
    for i in range(max_threads):
        print("Tread ", i + 1, " is running")
        chunk_size = (size // max_threads) if i < max_threads - 1 else (size // max_threads) + (size % max_threads)
        t = Thread(target=generate_and_save_streets,
                   args=(address, opt.backgrounds, opt.cars,
                         chunk_size, tuple(opt.img_size), tuple(opt.grid_size)))
        t.start()
        threadList.append(t)
        if i == 0:
            time.sleep(4)
    for t in threadList:
        t.join()
