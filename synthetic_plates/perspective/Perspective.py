import numpy as np
import cv2
import functools
import random
from PIL import Image

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from Utils import set_background


def compare(rect1, rect2):
    if abs(rect1[1] - rect2[1]) > 10:
        return rect1[1] - rect2[1]
    else:
        return rect1[0] - rect2[0]


def get_mask(img):
    # img[np.where(img != 255)] = 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 45)
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
        mask = cv2.add(mask, labelMask)
        # if numPixels > 5 and numPixels < 800:
        #     mask = cv2.add(mask, labelMask)

    return mask


def get_bounding_boxes(img):
    mask = get_mask(img)

    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes1 = [cv2.boundingRect(c) for c in cnts]
    boundingBoxes = sorted(boundingBoxes1, key=functools.cmp_to_key(compare))

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
            merged_boxes.append([x - 3, y - 3, w + 6, h + 6])

    return merged_boxes, boundingBoxes


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


def create_perspective(img, mask, img_size: tuple, noises: list, prespectiveType: int = 0,
                       pad: tuple = (100, 100, 100, 100)):
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
    mask = np.array(mask)[:, :, :-1][:, :, ::-1]

    merged_boxes, _ = get_bounding_boxes(mask)

    before_altering = img.copy()

    # Apply noises
    for noise in noises:
        before_altering = noise.apply(before_altering)

    before_altering, mask, merged_boxes = set_background(before_altering, mask, merged_boxes, img_size[0], img_size[1])

    height, width = before_altering.shape[:2]
    mask = cv2.copyMakeBorder(mask, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    before_altering = cv2.copyMakeBorder(before_altering, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT)
    merged_boxes[:, 0] = merged_boxes[:, 0] + pad[2]
    merged_boxes[:, 1] = merged_boxes[:, 1] + pad[0]

    pType = prespectiveType
    if prespectiveType == 0:
        pType = np.random.randint(1, 7)
    matrix = get_perspective_matrix(width, height, pType)
    newHeight, newWidth = mask.shape[:2]

    imgOutput = mask.copy()

    imgOutput = cv2.warpPerspective(imgOutput, matrix, (newWidth, newHeight),
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))
    before_altering = cv2.warpPerspective(before_altering, matrix, (newWidth, newHeight),
                                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0))

    up_left_boxes = np.array([[x, y, 1] for (x, y, w, h) in merged_boxes]).T
    down_right_boxes = np.array([[x + w, y + h, 1] for (x, y, w, h) in merged_boxes]).T
    up_right_boxes = np.array([[x + w, y, 1] for (x, y, w, h) in merged_boxes]).T
    down_left_boxes = np.array([[x, y + h, 1] for (x, y, w, h) in merged_boxes]).T

    up_left_boxes = matrix @ up_left_boxes
    down_right_boxes = matrix @ down_right_boxes
    up_right_boxes = matrix @ up_right_boxes
    down_left_boxes = matrix @ down_left_boxes
    boxes = []
    for (ul, dr, ur, dl) in zip(up_left_boxes.T, down_right_boxes.T, up_right_boxes.T, down_left_boxes.T):
        x = int(np.min([ul[0], dr[0], ur[0], dl[0]]))
        y = int(np.min([ul[1], dr[1], ur[1], dl[1]]))
        w = int(np.max([ul[0], dr[0], ur[0], dl[0]]) - np.min([ul[0], dr[0], ur[0], dl[0]]))
        h = int(np.max([ul[1], dr[1], ur[1], dl[1]]) - np.min([ul[1], dr[1], ur[1], dl[1]]))
        boxes.append((x, y, w, h))

    return (before_altering, imgOutput, boxes)
