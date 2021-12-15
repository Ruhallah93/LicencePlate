from .Noise import Noise
import cv2
from PIL import Image
import numpy as np


class ImageNoise(Noise):
    def __init__(self, pathToImage):
        super()
        self.pathToImage = pathToImage

    def apply(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        im_pil = Image.fromarray(img)
        noise = Image.open(self.pathToImage).convert("RGBA").resize((312, 70))
        im_pil.paste(noise, (0, 0), mask=noise)

        open_cv_image = np.array(im_pil)[:, :, :-1]
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        return open_cv_image
