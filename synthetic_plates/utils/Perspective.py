import numpy as np
import cv2
import functools

from .Utils import set_background


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


def get_rotate_matrix(x, y, angle):
    angle = np.deg2rad(angle)  # degree to radian
    move_matrix = np.array(
        [
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]
        ])
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    back_matrix = np.array(
        [
            [1, 0, -x],
            [0, 1, -y],
            [0, 0, 1]
        ])

    return (move_matrix @ rotation_matrix) @ back_matrix


def roll(p1, p2, p3, p4, x, y, theta):
    rotate = get_rotate_matrix(x, y, theta)
    p = [[p1[0], p2[0], p3[0], p4[0]],
         [p1[1], p2[1], p3[1], p4[1]],
         [1, 1, 1, 1]]
    p1, p2, p3, p4 = (rotate @ p).T[:, :2]
    return p1, p2, p3, p4


def pitch(p1, p2, p3, p4, type, const_h):
    if type == 1:
        p1 = (p1[0] - const_h, p1[1] + const_h)
        p2 = (p2[0] + const_h, p2[1] + const_h)
        p3 = (p3[0] - const_h, p3[1] - const_h)
        p4 = (p4[0] + const_h, p4[1] - const_h)
    else:
        p1 = (p1[0] + const_h, p1[1] + const_h)
        p2 = (p2[0] - const_h, p2[1] + const_h)
        p3 = (p3[0] + const_h, p3[1] - const_h)
        p4 = (p4[0] - const_h, p4[1] - const_h)
    return p1, p2, p3, p4


def yaw(p1, p2, p3, p4, type, const_h):
    if type == 1:
        p1 = (p1[0] + const_h, p1[1] - const_h)
        p2 = (p2[0] - const_h, p2[1] + const_h)
        p3 = (p3[0] - const_h, p3[1] - const_h)
        p4 = (p4[0] + const_h, p4[1] + const_h)
    else:
        p1 = (p1[0] + const_h, p1[1] + const_h)
        p2 = (p2[0] - const_h, p2[1] - const_h)
        p3 = (p3[0] - const_h, p3[1] + const_h)
        p4 = (p4[0] + const_h, p4[1] - const_h)
    return p1, p2, p3, p4


def get_perspective_matrix(width, height, perspective_type, pad: tuple = (100, 100, 100, 100)):
    """
    get prespective matrix for an image
    """
    p1, p2, p3, p4 = ((pad[2], pad[0]),
                      (pad[2] + width - 1, pad[0]),
                      (pad[2] + width - 1, pad[0] + height - 1),
                      (pad[2], pad[0] + height - 1))
    inp = np.float32([[p1[0], p1[1]],
                      [p2[0], p2[1]],
                      [p3[0], p3[1]],
                      [p4[0], p4[1]]])

    direction = np.random.choice([1, -1])
    theta = np.random.randint(0, 15) * direction
    step_length = np.random.randint(1, 30)
    if perspective_type == 1:
        p1, p2, p3, p4 = pitch(p1, p2, p3, p4, type=direction, const_h=step_length)
    if perspective_type == 2:
        p1, p2, p3, p4 = yaw(p1, p2, p3, p4, type=direction, const_h=step_length)
    if perspective_type == 3:
        p1, p2, p3, p4 = roll(p1, p2, p3, p4, x=width / 2, y=height / 2, theta=theta)
    if perspective_type == 4:
        p1, p2, p3, p4 = pitch(p1, p2, p3, p4, type=direction, const_h=step_length)
        step_length = np.random.randint(1, 30)
        p1, p2, p3, p4 = yaw(p1, p2, p3, p4, type=direction, const_h=step_length)
    if perspective_type == 5:
        p1, p2, p3, p4 = pitch(p1, p2, p3, p4, type=direction, const_h=step_length)
        step_length = np.random.randint(1, 30)
        p1, p2, p3, p4 = yaw(p1, p2, p3, p4, type=direction, const_h=step_length)
        p1, p2, p3, p4 = roll(p1, p2, p3, p4, x=width / 2, y=height / 2, theta=theta)

    out = np.float32([[p1[0], p1[1]],
                      [p2[0], p2[1]],
                      [p3[0], p3[1]],
                      [p4[0], p4[1]]])

    return cv2.getPerspectiveTransform(inp, out)


# def get_perspective_matrix(width, height, pad: tuple = (100, 100, 100, 100)):
#     """
#     get prespective matrix for an image
#     """
#     p1, p2, p3, p4 = ((pad[2], pad[0]),
#                       (pad[2] + width - 1, pad[0]),
#                       (pad[2] + width - 1, pad[0] + height - 1),
#                       (pad[2], pad[0] + height - 1)
#                       )
#     inp = np.float32([[p1[0], p1[1]],
#                       [p2[0], p2[1]],
#                       [p3[0], p3[1]],
#                       [p4[0], p4[1]]])
#
#     if prespectiveType == 1:
#         p2 = (p2[0], p2[1] - const_h)
#         p3 = (p3[0], p3[1] - const_h)
#     elif prespectiveType == 2:
#         p2 = (p2[0], p2[1] + const_h)
#         p3 = (p3[0], p3[1] + const_h)
#     elif prespectiveType == 3:
#         p1 = (p1[0], p1[1] - const_h)
#         p4 = (p4[0], p4[1] - const_h)
#     elif prespectiveType == 4:
#         p1 = (p1[0], p1[1] + const_h)
#         p4 = (p4[0], p4[1] + const_h)
#     elif prespectiveType == 5:
#         p2 = (p2[0], p2[1] - const_h)
#         p3 = (p3[0], p3[1] - const_h)
#
#         p1 = (p1[0], p1[1] + const_h)
#         p4 = (p4[0], p4[1] + const_h)
#     elif prespectiveType == 6:
#         p2 = (p2[0], p2[1] + const_h)
#         p3 = (p3[0], p3[1] + const_h)
#
#         p1 = (p1[0], p1[1] - const_h)
#         p4 = (p4[0], p4[1] - const_h)
#     else:
#         return None
#
#     out = np.float32([[p1[0], p1[1]],
#                       [p2[0], p2[1]],
#                       [p3[0], p3[1]],
#                       [p4[0], p4[1]]])
#
#     return cv2.getPerspectiveTransform(inp, out)

def after_transform(p, matrix):
    px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
        (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    return [int(px), int(py), 1]


def create_perspective(img, mask, img_size: tuple, noises: list, perspective_type: int = 1,
                       pad: tuple = (100, 100, 100, 100)):
    """
    This function applies perspective to the image with the given path

    Keyword arguments:
    pathToImage -- path to the image file in png format (string)
    perspectiveType -- type of perspective (integer between 1 to 6), others mean no perspective
            would be applied
    pad -- padding for the image (top,bottom,left,right)

    returns -- a tuple: (altered image with perspective good for bounding box extraction, original image that
                            with paddings and same perspective)
    """

    # if not pathToImage.endswith('.png'):
    #     raise Exception('Only png files are supported.')
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

    matrix = get_perspective_matrix(width, height, perspective_type, pad=(0, 0, 0, 0))
    newHeight, newWidth = mask.shape[:2]

    imgOutput = mask.copy()

    imgOutput = cv2.warpPerspective(imgOutput, matrix, (newWidth, newHeight),
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))
    before_altering = cv2.warpPerspective(before_altering, matrix, (newWidth, newHeight),
                                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(0, 0, 0))

    up_left_boxes = np.array([after_transform([x, y], matrix) for (x, y, w, h) in merged_boxes]).T
    down_right_boxes = np.array([after_transform([x + w, y + h], matrix) for (x, y, w, h) in merged_boxes]).T
    up_right_boxes = np.array([after_transform([x + w, y], matrix) for (x, y, w, h) in merged_boxes]).T
    down_left_boxes = np.array([after_transform([x, y + h], matrix) for (x, y, w, h) in merged_boxes]).T

    boxes = []
    for (ul, dr, ur, dl) in zip(up_left_boxes.T, down_right_boxes.T, up_right_boxes.T, down_left_boxes.T):
        x = int(np.min([ul[0], dr[0], ur[0], dl[0]]))
        y = int(np.min([ul[1], dr[1], ur[1], dl[1]]))
        w = int(np.max([ul[0], dr[0], ur[0], dl[0]]) - np.min([ul[0], dr[0], ur[0], dl[0]]))
        h = int(np.max([ul[1], dr[1], ur[1], dl[1]]) - np.min([ul[1], dr[1], ur[1], dl[1]]))
        boxes.append((x, y, w, h))

    return (before_altering, imgOutput, boxes)
