import uuid

from threading import Thread
import argparse
import cv2
import random
from PIL import Image
import os
from utils.Utils import letter_to_class
from datetime import datetime
from utils.Creator import get_new_plate


def generate_and_save_plates(address, dataset_size: int = 200, img_size: tuple = (600, 400),
                             save_bounding_boxes=True, save_mask=True, mask_state='grayscale'):
    random.seed(datetime.now())

    counter = 0
    for i in range(dataset_size):
        plate, perspective_plate, mask, bonding_boxes = get_new_plate(img_size, mask_state=mask_state)
        if save_bounding_boxes:
            if len(bonding_boxes) != 8:
                counter += 1
                print("len(merged_boxes): ", len(bonding_boxes))
                for box in bonding_boxes:
                    x, y, w, h = box
                    cv2.rectangle(perspective_plate, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.imshow('perspective_plate', perspective_plate)
                cv2.imshow('mask', mask)
                cv2.waitKey()
                continue

        perspective_plate = cv2.cvtColor(perspective_plate, cv2.COLOR_BGR2RGBA)
        perspective_plate = Image.fromarray(perspective_plate)

        _id = uuid.uuid4().__str__()
        name = plate[0] + plate[1] + '_' + plate[2] + '_' + plate[3] + plate[4] + plate[5] + plate[6] + plate[7]

        separetor = os.sep

        address = address + separetor if address[-1] != separetor else address
        directory = address + "images" + separetor if save_mask else address
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save bounding boxes
        if save_bounding_boxes:
            label_file = open("{}.txt".format(directory + name + '$' + _id + "txt"), 'w')
            height, width = perspective_plate.height, perspective_plate.width
            classes = [plate[0], plate[1], letter_to_class[plate[2]], plate[3], plate[4], plate[5], plate[6],
                       plate[7]]
            for i, box in enumerate(sorted(bonding_boxes, key=lambda x: x[0])):
                x, y, w, h = box
                x_center = int((x + (0.5) * w)) / width
                y_center = int((y + (0.5) * h)) / height
                label_file.write("{} {} {} {} {}\n".format(classes[i], x_center, y_center, w / width, h / height))
            label_file.close()

        # Save plates
        perspective_plate.save(directory + name + '$' + _id + ".png")

        # Save masks
        if save_mask:
            directory = address + "masks" + separetor
            if not os.path.exists(directory):
                os.makedirs(directory)
            masked = Image.fromarray(mask)
            masked.save(directory + name + '$' + _id + ".png")

    print("fails: ", counter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=2000, help='number of plates to generate')
    parser.add_argument('--workers', type=int, default=10, help='number of threads to run')
    parser.add_argument('--img_size', nargs='+', type=int, default=[500, 400], help='size of background')
    parser.add_argument('--save_bounding_boxes', action='store_true', help='save the bounding boxes if true')
    parser.add_argument('--save_mask', action='store_true', help='save the masks if true')
    parser.add_argument('--mask_state', type=str, default='grayscale', help='grayscale or colorful')
    parser.add_argument('--address', type=str, default='output/example3', help='The address of saving dataset')
    opt = parser.parse_args()
    threadList = []

    # opt.save_mask = False
    # opt.save_bounding_boxes = True
    # opt.mask_state = "grayscale"

    address = opt.address + os.sep if opt.address[-1] != os.sep else opt.address
    directory = address + "images" + os.sep if opt.save_mask else address

    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = address + "masks" + os.sep if opt.save_mask else address
    if not os.path.exists(directory):
        os.makedirs(directory)

    # if not os.path.exists(opt.address):
    #     os.makedirs(opt.address)

    size = opt.size
    max_threads = opt.workers
    for i in range(max_threads):
        chunk_size = (size // max_threads) if i < max_threads - 1 else (size // max_threads) + (size % max_threads)
        t = Thread(target=generate_and_save_plates, args=(opt.address, chunk_size, tuple(opt.img_size),
                                                          opt.save_bounding_boxes, opt.save_mask, opt.mask_state))
        t.start()
        threadList.append(t)
    for t in threadList:
        t.join()
