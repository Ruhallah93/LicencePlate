import glob
import os
import numpy as np
import shutil
from PIL import Image
import cv2

letter_to_class = {"ALEF": 10, "BE": 11, "PE": 12, "TE": 13, "SE": 14, "JIM": 15, "CHE": 16, "HEY": 17, "KHE": 18,
                   "DAL": 19, "ZAL": 20, "RE": 21, "ZE": 22, "ZHE": 23, "SIN": 24, "SHIN": 25, "SAD": 26, "ZAD": 27,
                   "TA": 28, "ZA": 29, "EIN": 30, "GHEIN": 31, "FE": 32, "GHAF": 33, "KAF": 34, "GAF": 35, "LAM": 36,
                   "MIM": 37, "NON": 38, "VAV": 39, "HE": 40, "YE": 41, "WHEEL": 42, "DPLMT": 43, "SYSI": 44,
                   "TSHFT": 45}
class_to_letter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "ALEF", "BE", "PE", "TE", "SE", "JIM", "CHE", "HEY", "KHE", "DAL",
                   "ZAL", "RE", "ZE", "ZHE", "SIN", "SHIN", "SAD", "ZAD", "TA", "ZA", "EIN", "GHEIN", "FE", "GHAF",
                   "KAF", "GAF", "LAM", "MIM", "NON", "VAV", "HE", "YE", "WHEEL", "DPLMT", "SYSI", "TSHFT"]

show_class_to_letter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "AL", "BE", "PE", "TE", "SE", "JIM", "CHE", "HEY", "KHE", "DAL",
                        "ZAL", "RE", "ZE", "ZHE", "SIN", "SH", "SAD", "ZAD", "TA", "ZA", "EIN", "GHEIN", "FE", "GHAF",
                        "KAF", "GAF", "LAM", "MIM", "NON", "VAV", "HE", "YE", "WHE", "D", "S", "TSHFT"]


class LabelCheck:
    # SHOW_IMAGE = 0
    # SHOW_LABEL = 1
    # SHOW_BORDER = 2
    # SHOW_BORDER_LABEL = 3

    def __init__(self, dataset_path, def_func=True, show_first=True, label_show=True, border_show=True,
                 win_width=400, win_height=200):
        super()
        self.window_name = "window"
        self.dataset_path = dataset_path
        self.wrong_path = ''
        self.quality_path = ''
        self.text_list = glob.glob(self.dataset_path + "*.txt")
        self.counter = -1

        self.functions = {}

        self.label_show = label_show
        self.border_show = border_show
        self.win_width = win_width
        self.win_height = win_height

        if def_func:
            self.next_image_func()
            self.prev_image_func()
            self.label_show_func()
            self.border_show_func()
            self.move_wrong_func()
            self.move_quality_func()

        cv2.namedWindow(self.window_name)

        if show_first:
            self.show_next_image()

    ############################
    # Get

    def get_text_path(self):
        return f"{self.dataset_path + os.path.basename(self.text_list[self.counter])}"

    def get_image_path(self):
        return f"{self.dataset_path + os.path.basename(self.text_list[self.counter]).split('.')[0]}.jpg"

    ############################
    # Change object attributes

    def set_path(self, path_name, path):
        if path_name == 'wrong':
            self.wrong_path = path
        if path_name == 'quality':
            self.quality_path = path

    ############################
    # Key binding functions

    def get_key_code(self):
        while True:
            key = cv2.waitKey()
            print(key)
            if key == 27:
                break

    def key_binding(self, key_code, function):
        self.functions[str(key_code)] = function

    ############################
    # Label convert functions

    def text_to_label(self, line):
        l, x, y, w, h = np.array(line.split(' ')).astype(float)
        # _, _, ww, wh = cv2.getWindowImageRect(self.window_name)
        ww, wh = self.win_width, self.win_height
        x = int(x * ww)
        y = int(y * wh)
        w = int(w * ww)
        h = int(h * wh)

        return show_class_to_letter[int(l)], x, y, w, h

    def label_to_points(self, labels):
        x, y, w, h = labels

        return (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2))

    ############################
    # Show Image function

    def show_image(self, image):
        # resize
        res = cv2.resize(image, (self.win_width, self.win_height))
        # RGB2BGR
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

        if self.label_show and not self.border_show:
            cv2.putText(res, os.path.basename(self.text_list[self.counter]).split('.')[0], (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv2.LINE_AA)

        if not self.label_show and self.border_show:
            text_path = self.get_text_path()
            f = open(text_path, "r")
            for line in f:
                l, x, y, w, h = self.text_to_label(line)
                p1, p2 = self.label_to_points((x, y, w, h))

                cv2.rectangle(res, p1, p2, (36, 255, 12), 2)
            f.close()

        if self.label_show and self.border_show:
            text_path = self.get_text_path()
            f = open(text_path, "r")
            for line in f:
                l, x, y, w, h = self.text_to_label(line)
                p1, p2 = self.label_to_points((x, y, w, h))

                cv2.rectangle(res, p1, p2, (36, 255, 12), 2)
                cv2.putText(res, str(l), p1, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            f.close()

        cv2.imshow(self.window_name, res)

    ############################
    # Forward & Backward images

    def show_current_image(self):
        image_path = self.get_image_path()
        image = np.array(Image.open(image_path))
        self.show_image(image)

    def show_next_image(self):
        self.counter += 1
        if self.counter < len(self.text_list):
            self.show_current_image()
        else:
            print('end of files')
            self.counter -= 1

    def next_image_func(self, key_code=100):  # D: 100
        self.key_binding(key_code, self.show_next_image)

    def show_prev_image(self):
        self.counter -= 1
        if self.counter >= 0:
            self.show_current_image()
        else:
            print('start of files')
            self.counter += 1

    def prev_image_func(self, key_code=97):  # A: 97
        self.key_binding(key_code, self.show_prev_image)

    ############################
    # Display option

    def change_display_label(self):
        self.label_show = not self.label_show
        self.show_current_image()

    def label_show_func(self, key_code=108):  # L: 108
        self.key_binding(key_code, self.change_display_label)

    def change_display_border(self):
        self.border_show = not self.border_show
        self.show_current_image()

    def border_show_func(self, key_code=98):  # B: 98
        self.key_binding(key_code, self.change_display_border)

    ############################
    # Move Incorrect & Bad quality images

    def move_image(self, des):
        text_path = self.get_text_path()
        image_path = self.get_image_path()
        shutil.move(text_path, des)
        shutil.move(image_path, des)
        self.text_list.pop(self.counter)
        self.show_current_image()

    def move_wrong(self):
        self.move_image(self.wrong_path)

    def move_wrong_func(self, key_code=119):  # W: 119
        self.key_binding(key_code, self.move_wrong)

    def quality_func(self):
        self.move_image(self.quality_path)

    def move_quality_func(self, key_code=113):  # Q: 113
        self.key_binding(key_code, self.quality_func)

    ############################
    # Start showing

    def start(self):
        while True:
            key = cv2.waitKey()
            if key == 27:
                break
            for i in self.functions.keys():
                if int(i) == key:
                    self.functions[i]()


dataset_path = "/home/ruhiii/Downloads/CharFinder/Tasks/Rouhollah-All/Rouhollah/"
wrong_path = "/home/ruhiii/Downloads/CharFinder/Tasks/Rouhollah-All/Rouhollah_Incorrect/"
quality_path = "/home/ruhiii/Downloads/CharFinder/Tasks/Rouhollah-All/Rouhollah_Ambihuous/"

labelCheck = LabelCheck(dataset_path)
labelCheck.set_path('wrong', wrong_path)
labelCheck.set_path('quality', quality_path)
# labelCheck.get_key_code()
labelCheck.start()
