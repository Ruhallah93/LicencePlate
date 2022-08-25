from .Noise import Noise
import cv2
import numpy as np


class LightNoise(Noise):
    # light_param: [-255, 255], light noise parameter
    # area: -1 in part 2 means go to end, Example: area=[[0, 0], [-1, 40]]
    def __init__(self, blur_kernel_size=0, light_param=100, area=-1):
        super()
        self.blur_kernel_size = int(np.round(blur_kernel_size))
        self.light_param = int(np.round(light_param))
        self.area = area

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
            y_noise = np.array(yuv_img[:, :, 0], dtype=np.int16)
            y_noise[self.area[0][1]: self.area[1][1], self.area[0][0]:self.area[1][0]] += self.light_param
        else:
            # Add noise to Y channel (Y_channel = yuv_img[:, :, 0])
            # int16 for support overflow and negative values
            y_noise = np.array(yuv_img[:, :, 0], dtype=np.int16) + self.light_param

        # Overflow handling
        if self.light_param > 0:
            y_noise[y_noise > 255] = 255
        else:
            y_noise[y_noise < 0] = 0

        y_noise = np.array(y_noise, dtype=np.uint8)
        yuv_img[:, :, 0] = y_noise

        # Remove, just for test
        # cv2.imshow("d", self.YUV2RGBA(yuv_img))
        # cv2.waitKey()

        return self.YUV2RGBA(yuv_img)
