import cv2

from .Noise import Noise
import numpy as np

class BlurNoise(Noise):
    def __init__(self, blur_type='gaussian', blur_kernel_size=7, blur_sigma=10):
        super()
        self.blur_type = blur_type
        self.blur_kernel_size = int(np.round(blur_kernel_size))
        self.blur_sigma = int(np.round(blur_sigma))

    def apply(self, img):
        # cv2.imshow("s", img)

        # print(img.shape)


        # kernel size checking
        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1
            print("blur_kernel_size for medianBlur should be a odd number, Alternative kernel_size: ",
                  self.blur_kernel_size)

        if self.blur_type == 'median':
            res = cv2.medianBlur(img, ksize=self.blur_kernel_size)
        elif self.blur_type == 'gaussian':
            res = cv2.GaussianBlur(img, ksize=(self.blur_kernel_size, self.blur_kernel_size), sigmaX=self.blur_sigma)

        # cv2.imshow('dd', res)
        # cv2.waitKey()

        return res
