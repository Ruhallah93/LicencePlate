from .Noise import Noise
import numpy as np


class SPNoise(Noise):
    def __init__(self, s_vs_p=0.5, amount=0.1, bw=True, salt_color=255, pepper_color=0):
        super()
        self.s_vs_p = s_vs_p
        self.amount = amount
        self.bw = bw
        self.salt_color = salt_color
        self.pepper_color = pepper_color

    def apply(self, img):
        # cv2.imshow("s", img)

        out = np.copy(img)

        # Salt mode
        num_salt = np.ceil(self.amount * img.size * self.s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt))
                  for i in img.shape]

        if self.bw:
            out[coords[0], coords[1], :] = self.salt_color
        else:
            out[coords] = self.salt_color

        # Pepper mode
        num_pepper = np.ceil(self.amount * img.size * (1. - self.s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper))
                  for i in img.shape]
        if self.bw:
            out[coords[0], coords[1], :] = self.pepper_color
        else:
            out[coords] = self.pepper_color

        # cv2.imshow('end', out)
        # cv2.waitKey()

        return out
