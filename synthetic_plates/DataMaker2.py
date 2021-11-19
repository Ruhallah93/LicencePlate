import uuid

from threading import Thread
import argparse

import numpy as np
import cv2
import functools

from enum import Enum
import numpy as np
import cv2
from PIL import Image
import functools
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
import random
from PIL import Image
import os
from datetime import datetime

# Characters of Letters and Numbers in Plates
numbers = [str(i) for i in range(0, 10)]
letters = ["BE", "TE", "JIM", "DAL", "RE", "SIN", "SAD", "TA", "EIN", "GHAF", "LAM", "MIM", "NON", "VAV", "HE",
           "YE", "WHEEL"]
letter_to_class = {"ALEF": 10, "BE": 11, "PE": 12, "TE": 13, "SE": 14, "JIM": 15, "CHE": 16, "HEY": 17, "KHE": 18,
                   "DAL": 19, "ZAL": 20, "RE": 21, "ZE": 22, "ZHE": 23,
                   "SIN": 24, "SHIN": 25, "SAD": 26, "ZAD": 27, "TA": 28, "ZA": 29, "EIN": 30, "GHEIN": 31, "FE": 32,
                   "GHAF": 33, "KAF": 34, "GAF": 35, "LAM": 36, "MIM": 37, "NON": 38,
                   "VAV": 39, "HE": 40, "YE": 41, "WHEEL": 42}


class Noise(object):
    """
    Each noise should inherite this class and have exact methods of this class
    """

    def __init__(self):
        # init
        pass

    def apply(self, img):
        """
        this function gets an image and applies the corresponding noise of class on it
        img: -- cv2 image

        return: an image that the noise is applied on
        """
        pass
        # return an image that the noise is applied on


class ImageNoise(Noise):
    def __init__(self, pathToImage):
        super()
        self.pathToImage = pathToImage

    def apply(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        im_pil = Image.fromarray(img)
        noise = Image.open(self.pathToImage).convert("RGBA")
        im_pil.paste(noise, (0, 0), mask=noise)

        open_cv_image = np.array(im_pil)[:, :, :-1]
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        return open_cv_image


class LightNoise(Noise):
    # r is degree of light
    def __init__(self, r):
        super()
        self.r = r

    def apply(self, img):
        # TODO: Dear Bideh, please add light noise to the img
        return img


def get_perspective_matrix(width, height, prespectiveType: int = 1,
                           pad: tuple = (100, 100, 100, 100), const_h=30, const_w=15):
    """
    get prespective matrix for an image
    """
    p1, p2, p3, p4 = ((pad[2], pad[0]),
                      (pad[2] + width - 1, pad[0]),
                      (pad[2] + width - 1, pad[0] + height - 1),
                      (pad[2], pad[0] + height - 1)
                      )
    inp = np.float32([[p1[0], p1[1]],
                      [p2[0], p2[1]],
                      [p3[0], p3[1]],
                      [p4[0], p4[1]]])

    if prespectiveType == 1:
        p2 = (p2[0], p2[1] - const_h)
        p3 = (p3[0], p3[1] - const_h)
    elif prespectiveType == 2:
        p2 = (p2[0], p2[1] + const_h)
        p3 = (p3[0], p3[1] + const_h)
    elif prespectiveType == 3:
        p1 = (p1[0], p1[1] - const_h)
        p4 = (p4[0], p4[1] - const_h)
    elif prespectiveType == 4:
        p1 = (p1[0], p1[1] + const_h)
        p4 = (p4[0], p4[1] + const_h)
    elif prespectiveType == 5:
        p2 = (p2[0], p2[1] - const_h)
        p3 = (p3[0], p3[1] - const_h)

        p1 = (p1[0], p1[1] + const_h)
        p4 = (p4[0], p4[1] + const_h)
    elif prespectiveType == 6:
        p2 = (p2[0], p2[1] + const_h)
        p3 = (p3[0], p3[1] + const_h)

        p1 = (p1[0], p1[1] - const_h)
        p4 = (p4[0], p4[1] - const_h)
    else:
        return None

    out = np.float32([[p1[0], p1[1]],
                      [p2[0], p2[1]],
                      [p3[0], p3[1]],
                      [p4[0], p4[1]]])

    return cv2.getPerspectiveTransform(inp, out)


def create_perspective(img, noises: list, prespectiveType: int = 0, pad: tuple = (100, 100, 100, 100)):
    """
    This function applies prespective to the image with the given path

    Keyword arguments:
    pathToImage -- path to the image file in png format (string)
    prespectiveType -- type of prespective (integer between 0 to 6), if 0 acts random o.w a specific prespective
            would be applied
    pad -- padding for the image (top,bottom,left,right)

    returns -- a tuple: (altered image with prespective good for bounding box extraction, original image that
                            with paddings and same prespective)
    """

    # if not pathToImage.endswith('.png'):
    #     raise Exception('Only png files are supported.')
    if type(prespectiveType) != int or prespectiveType not in range(0, 7):
        raise Exception('prespectiveType argument must be and integer between 1 to 6.')
    if type(pad) != tuple or len(pad) != 4:
        raise Exception('pad argument must be a tuple of size 4 with integers inside')

    # img = cv2.imread(pathToImage)
    img = np.array(img)[:, :, :-1][:, :, ::-1]
    before_altering = img.copy()

    for noise in noises:
        before_altering = noise.apply(before_altering)

    bg = img[10, 35, :]
    img[:, :32, :] = bg
    img[:, 235:242, :] = bg
    img[:6, :, :] = bg
    img[:, -6:, :] = bg
    img[-5:, :, :] = bg
    img[:20, 242:300, :] = bg
    height, width = img.shape[:2]
    img = cv2.copyMakeBorder(img, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT)
    before_altering = cv2.copyMakeBorder(before_altering, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT)
    pType = prespectiveType
    if prespectiveType == 0:
        pType = np.random.randint(1, 7)
    matrix = get_perspective_matrix(width, height, pType)
    newHeight, newWidth = img.shape[:2]
    imgOutput = cv2.warpPerspective(img, matrix, (newWidth, newHeight),
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))
    before_altering = cv2.warpPerspective(before_altering, matrix, (newWidth, newHeight),
                                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0))
    return (before_altering, imgOutput)


def compare(rect1, rect2):
    if abs(rect1[1] - rect2[1]) > 10:
        return rect1[1] - rect2[1]
    else:
        return rect1[0] - rect2[0]


def get_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")
    for (i, label) in enumerate(np.unique(labels)):
        # If this is the background label, ignore it
        if label == 0:
            continue

        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        # If the number of pixels in the component is between lower bound and upper bound,
        # add it to our mask
        if numPixels > 5 and numPixels < 800:
            mask = cv2.add(mask, labelMask)
    return mask   

def get_bounding_boxes(img):
    mask = get_mask(img)

    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))

    merged_boxes = list()
    banned = []
    for i, box in enumerate(boundingBoxes):
        x, y, w, h = box
        if i not in banned:
            for j, box2 in enumerate(boundingBoxes):
                x2, y2, w2, h2 = box2
                if i != j and ((x2 >= x and x2 + w2 <= x + w) or (x >= x2 and x + w <= x2 + w2)):
                    upper = np.min([y, y2])
                    lower = np.max([y2 + h2, y + h])
                    h = lower - upper
                    y = upper
                    banned.append(j)
            merged_boxes.append((x, y, w, h))
    return merged_boxes, boundingBoxes


# Returns an array containing a plate's letter and numbers:
# [number1, number2 , letter, number3, number4, number5]
def get_new_plate_number():
    return [random.choice(numbers),
            random.choice(numbers),
            random.choice(letters),
            random.choice(numbers),
            random.choice(numbers),
            random.choice(numbers),
            random.choice(numbers),
            random.choice(numbers)]


# Returns Address of a glyph image given font, and glyph name
def get_glyph_address(glyph_name):
    return os.path.join("./Glyphs/b_roya", "{}.png".format(glyph_name))


def get_new_plate():
    plate = get_new_plate_number()

    # Get Glyph images of plate characters
    glyph_images = [Image.open(get_glyph_address(glyph)).convert("RGBA") for glyph in plate]

    # Create a blank image with size of templates
    # and add the background and glyph images
    new_plate = Image.new('RGBA', (600, 132), (0, 0, 0, 0))

    if plate[2] in ["TE", "EIN"]:
        background = Image.open("./templates/template-ommomi.png").convert("RGBA")
    else:
        background = Image.open("./templates/template-base.png").convert("RGBA")
    new_plate.paste(background, (0, 0))

    # adding glyph images with 11 pixel margin
    w = 0
    for i, glyph in enumerate(glyph_images[:-2]):
        if i == 2:
            new_plate.paste(glyph, (70 + w, 30), mask=glyph)
        else:
            new_plate.paste(glyph, (70 + w, 25), mask=glyph)
        w += glyph.size[0] + 3

    # last two digits
    w = 0
    for i, glyph in enumerate(glyph_images[-2:]):
        width, height = glyph.size[0], glyph.size[1]
        resized_glyph = glyph.resize((int(width * 0.75), int(height * 0.75)))
        new_plate.paste(resized_glyph, (485 + w, 50), mask=resized_glyph)
        w += glyph.size[0] - 10

    _newPlate = new_plate.resize((312, 70), Image.ANTIALIAS)

    imageNoise1 = ImageNoise('./noise/noise1.png')
    imageNoise2 = ImageNoise('./noise/noise2.png')
    imageNoise3 = ImageNoise('./noise/noise3.png')
    imageNoise4 = ImageNoise('./noise/noise4.png')
    imageNoise5 = ImageNoise('./noise/noise5.png')
    imageNoise6 = ImageNoise('./noise/noise6.png')
    noises1 = [imageNoise1, imageNoise2, imageNoise3, imageNoise4, imageNoise5, imageNoise6]

    imageNoise7 = ImageNoise('./noise/noise7.png')
    imageNoise8 = ImageNoise('./noise/noise8.png')
    lightNoise = LightNoise(random.randint(0, 100))
    noises2 = [imageNoise7, imageNoise8, lightNoise]

    r = random.randint(0, 3)
    noises = []
    if r == 1:
        noises = [random.choice(noises2 + noises1)]
    elif r == 2:
        noises = [random.choice(noises1), random.choice(noises2)]
    perspective_plate, for_bounding_boxes = create_perspective(_newPlate, noises=noises, pad=(50, 50, 10, 10))

    return plate, perspective_plate, for_bounding_boxes

def get_yolo_data():
    plate, perspective_plate, for_bounding_boxes = get_new_plate()
    ## get bounding boxes and plot them
    mergedBoxes, bb = get_bounding_boxes(for_bounding_boxes)

    return plate, perspective_plate, for_bounding_boxes, mergedBoxes

def get_unet_data():
    perspective_plate, for_bounding_boxes = get_new_plate()
    masked = get_mask(for_bounding_boxes)
    return plate, perspective_plate, masked

def generate_and_save_palets(n:int=1000):
    random.seed(datetime.now())
    
    counter = 0
    for i in range(n):
        plate, perspective_plate, for_bounding_boxes, merged_boxes = get_yolo_data()
        if len(merged_boxes) != 8:
            counter += 1
            print(len(merged_boxes))
            for box in merged_boxes:
                x, y, w, h = box
                cv2.rectangle(perspective_plate, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.imshow('123', perspective_plate)
            cv2.waitKey(1000)
            continue

        perspective_plate = cv2.cvtColor(perspective_plate, cv2.COLOR_BGR2RGBA)
        perspective_plate = Image.fromarray(perspective_plate)
        _id = uuid.uuid4().__str__()
        name = plate[0] + plate[1] + '_' + plate[2] + '_' + plate[3] + plate[4] + plate[5] + plate[6] + plate[7]
        perspective_plate.save('output/' + name + '$' + _id + ".png")

        label_file = open("{}.txt".format('output/' + name + '$' + _id + "txt"), 'w')
        height, width = perspective_plate.height, perspective_plate.width

        p1, p2, p3 = name.split("_")
        classes = [plate[0], plate[1], letter_to_class[plate[2]], plate[3], plate[4], plate[5], plate[6], plate[7]]
        for i, box in enumerate(sorted(merged_boxes, key=lambda x: x[0])):
            x, y, w, h = box
            x_center = int((x + (0.5) * w)) / width
            y_center = int((y + (0.5) * h)) / height
            label_file.write("{} {} {} {} {}\n".format(classes[i], x_center, y_center, w / width, h / height))

        label_file.close()
    print("fails: ", counter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', nargs='+', type=int, default=1000, help='number of plates to generate')
    parser.add_argument('--workers', nargs='+', type=int, default=10, help='number of threads to run')
    opt = parser.parse_args()
    
    size = opt.size[0]
    max_threads = opt.workers
    
    for i in range(max_threads):
        chunk_size = (size // max_threads) if i < max_threads - 1 else  (size // max_threads) + (size % max_threads) 
        t = Thread(target=generate_and_save_palets, args=[chunk_size])
        t.start()
    