from .Noise import Noise
import cv2
import numpy as np


class CircularLightNoise(Noise):
    # light_param: [-255, 255], light noise parameter
    # circular_light: number of circular lights
    def __init__(self, blur_kernel_size=0, light_param=100, n_circle=2, r_circle=25, kernel_sigma=0.7):
        super()
        self.blur_kernel_size = blur_kernel_size
        self.light_param = light_param
        self.n_circle = n_circle
        self.r_circle = r_circle
        self.kernel_sigma = kernel_sigma

    def gaussian_kernel(self, kernel_size=25, mu=0.0, sigma=1.0):
        x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
        d = np.sqrt(x * x + y * y)
        return np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

    def apply(self, img):
        # cv2.imshow("s", img)

        if self.blur_kernel_size > 0:
            if self.blur_kernel_size % 2 == 0:
                self.blur_kernel_size += 1
                print("blur_kernel_size for medianBlur should be a odd number, Alternative kernel_size: ",
                      self.blur_kernel_size)
            img = cv2.medianBlur(img, self.blur_kernel_size)

        # Change color space, BGR -> YUV
        yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # Add noise to Y channel (Y_channel = yuv_img[:, :, 0])
        height, width, _ = yuv_img.shape
        noise = np.zeros((height, width))

        # Create noise
        for i in range(self.n_circle):
            x = np.random.randint(0, width - self.r_circle * 2)
            y = np.random.randint(0, height - self.r_circle * 2)

            noise[y:y + self.r_circle * 2, x:x + self.r_circle * 2] += \
                self.gaussian_kernel(kernel_size=self.r_circle * 2, sigma=self.kernel_sigma) * self.light_param

        # int16 for support overflow and negative values
        y_noise = np.array(yuv_img[:, :, 0], dtype=np.int16) + noise

        # Overflow handling
        if self.light_param > 0:
            y_noise[y_noise > 255] = 255
        else:
            y_noise[y_noise < 0] = 0

        y_noise = np.array(y_noise, dtype=np.uint8)
        yuv_img[:, :, 0] = y_noise

        img_noise = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)

        # Remove, just for test
        # cv2.imshow("d", img_noise)
        # cv2.waitKey()

        return img_noise
