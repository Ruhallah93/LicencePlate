import numpy as np
import cv2


def get_rotate_matrix_with_center(x, y, angle):
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

    r = np.dot(move_matrix, rotation_matrix)
    return np.dot(r, back_matrix)


def roll(p1, p2, p3, p4, x, y, theta):
    rotate = get_rotate_matrix_with_center(x, y, theta)
    p1 = np.dot(rotate, [p1[0], p1[1], 1])[:2]
    p2 = np.dot(rotate, [p2[0], p2[1], 1])[:2]
    p3 = np.dot(rotate, [p3[0], p3[1], 1])[:2]
    p4 = np.dot(rotate, [p4[0], p4[1], 1])[:2]
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


def get_perspective_matrix(width, height, pad: tuple = (100, 100, 100, 100)):
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
    theta = np.random.randint(1, 10) * direction
    step_length = np.random.randint(1, 20)
    perspective_type = np.random.randint(1, 6)
    if perspective_type == 1:
        p1, p2, p3, p4 = pitch(p1, p2, p3, p4, type=direction, const_h=step_length)
    if perspective_type == 2:
        p1, p2, p3, p4 = yaw(p1, p2, p3, p4, type=direction, const_h=step_length)
    if perspective_type == 3:
        p1, p2, p3, p4 = roll(p1, p2, p3, p4, x=width / 2, y=height / 2, theta=theta)
    if perspective_type == 4:
        p1, p2, p3, p4 = pitch(p1, p2, p3, p4, type=direction, const_h=step_length)
        step_length = np.random.randint(1, 20)
        p1, p2, p3, p4 = yaw(p1, p2, p3, p4, type=direction, const_h=step_length)
    if perspective_type == 5:
        p1, p2, p3, p4 = pitch(p1, p2, p3, p4, type=direction, const_h=step_length)
        step_length = np.random.randint(1, 20)
        p1, p2, p3, p4 = yaw(p1, p2, p3, p4, type=direction, const_h=step_length)
        p1, p2, p3, p4 = roll(p1, p2, p3, p4, x=width / 2, y=height / 2, theta=theta)

    out = np.float32([[p1[0], p1[1]],
                      [p2[0], p2[1]],
                      [p3[0], p3[1]],
                      [p4[0], p4[1]]])

    return cv2.getPerspectiveTransform(inp, out)


plate_path = "/home/ruhiii/PycharmProjects/plate_recognition/synthetic_plates/files/templates/template-tashrifat.png"

img = cv2.imread(plate_path)

for type in [1, 2, 3, 4, 5]:
    height, width = img.shape[:2]
    matrix = get_perspective_matrix(width, height, pad=(0, 0, 0, 0))

    newHeight, newWidth = img.shape[:2]

    imgOutput = cv2.warpPerspective(img, matrix, (newWidth, newHeight),
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))

    cv2.imshow("img", img)
    cv2.imshow("imgOutput", imgOutput)
    cv2.waitKey()
