from .Noise import Noise
import cv2
from PIL import Image
import numpy as np


class ImageNoise(Noise):
    def __init__(self, pathToImage, plate_size: tuple = (312, 70)):
        super()
        self.pathToImage = pathToImage
        self.plate_size = plate_size

    def apply(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        im_pil = Image.fromarray(img)
        noise = Image.open(self.pathToImage).convert("RGBA").resize(self.plate_size)
        im_pil.paste(noise, (0, 0), mask=noise)

        # open_cv_image = np.array(im_pil)[:, :, :-1]
        # open_cv_image = open_cv_image[:, :, ::-1].copy()

        # open_cv_image = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGBA2BGRA)

        return np.array(im_pil)
