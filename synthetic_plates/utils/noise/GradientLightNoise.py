from .Noise import Noise
import cv2
import numpy as np


class GradientLightNoise(Noise):
    # r is degree of light [0:4]: 0, 90, 180, 270
    # light_param: [-255, 255], light noise parameter
    # area: -1 in part 2 means go to end, Example: area=[[0, 0], [-1, 40]]
    def __init__(self, blur_kernel_size=0, max_light_param=-170, area=-1, r=1):
        super()
        self.blur_kernel_size = blur_kernel_size
        self.max_light_param = max_light_param
        self.area = area
        self.r = r

    def apply(self, img):
        # cv2.imshow("s", img)

        if self.blur_kernel_size > 0:
            if self.blur_kernel_size % 2 == 0:
                self.blur_kernel_size += 1
                print("blur_kernel_size for medianBlur should be a odd number, Alternative kernel_size: ",
                      self.blur_kernel_size)
            img = cv2.medianBlur(img, self.blur_kernel_size)

        yuv_img = self.RGBA2YUV(img)

        if not self.area == -1:
            if self.area[1][0] == - 1:
                self.area[1][0] = img.shape[1]
            if self.area[1][1] == - 1:
                self.area[1][1] = img.shape[0]
        else:
            self.area = [[0, 0], [img.shape[1], img.shape[0]]]

        y_noise = np.array(yuv_img[:, :, 0], dtype=np.int16)
        noise = 0
        if self.r == 0:
            noise = np.array((self.area[1][1] - self.area[0][1]) * [
                np.arange(self.max_light_param - self.area[1][0] + self.area[0][0], self.max_light_param)])
        elif self.r == 2:
            noise = np.array((self.area[1][1] - self.area[0][1]) * [
                np.arange(self.max_light_param, self.max_light_param - self.area[1][0] + self.area[0][0], -1)])
        elif self.r == 1:
            noise = np.array((self.area[1][0] - self.area[0][0]) * [
                np.arange(self.max_light_param - self.area[1][1] + self.area[0][1],
                          self.max_light_param)]).transpose()
        elif self.r == 3:
            noise = np.array((self.area[1][0] - self.area[0][0]) * [
                np.arange(self.max_light_param, self.max_light_param - self.area[1][1] + self.area[0][1],
                          -1)]).transpose()
        ################################
        y_noise[self.area[0][1]: self.area[1][1], self.area[0][0]:self.area[1][0]] = y_noise[
                                                                                     self.area[0][1]: self.area[1][1],
                                                                                     self.area[0][0]:self.area[1][
                                                                                         0]] + noise

        # Overflow handling
        if self.max_light_param > 0:
            y_noise[y_noise > 255] = 255
        else:
            y_noise[y_noise < 0] = 0

        y_noise = np.array(y_noise, dtype=np.uint8)
        yuv_img[:, :, 0] = y_noise

        # Remove, just for test
        # cv2.imshow("d", img_noise)
        # cv2.waitKey()

        return self.YUV2RGBA(yuv_img)
