import cv2

class Noise(object):
    """
    Each noise should inherite this class and have exact methods of this class
    """

    def __init__(self):
        # init
        pass

    def apply(self, img):
        """
        this function gets an image and applies the corresponding noise of class on it
        img: -- cv2 image

        return: an image that the noise is applied on
        """
        pass
        # return an image that the noise is applied on

    def RGBA2YUV(self, img):
        # Change color space, RGBA -> RGB
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        # Change color space, RGB -> YUV
        yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

        return yuv_img

    def YUV2RGBA(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)

        # Convert BGR -> BGRA
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

        return img
