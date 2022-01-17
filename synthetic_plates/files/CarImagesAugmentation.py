import numpy as np
import imgaug as ia
from imgaug.augmentables.batches import UnnormalizedBatch
from imgaug import augmenters as iaa
import time
import cv2
import uuid
from PIL import Image
import glob
import os

separetor = os.sep
package_directory = os.path.dirname(os.path.abspath(__file__))
parent_path = package_directory[:package_directory.rindex(separetor)] + separetor
address = "files/cars/"
if not os.path.exists(parent_path + address):
    os.makedirs(parent_path + address)

images = []
root_dir = "files/cars_origin"

for filename in glob.glob(parent_path + root_dir + '/*.png'):
    im = Image.open(filename)
    images.append(np.array(im))
NB_BATCHES = 20

batches = [UnnormalizedBatch(images=images) for _ in range(NB_BATCHES)]

aug = iaa.Sometimes(0.5, [
    # iaa.PiecewiseAffine(scale=0.05, nb_cols=3, nb_rows=3),  # very slow
    iaa.Fliplr(0.5),  # very fast
    iaa.CropAndPad(px=(-20, 20)),  # very fast
    iaa.ChannelShuffle(0.35, channels=[0, 1, 2]),
    iaa.AdditiveGaussianNoise(scale=0.01 * 255),
    iaa.WithChannels([0, 1, 2], iaa.Add((10, 100))),
    iaa.RemoveCBAsByOutOfImageFraction(0.3),
    iaa.Sharpen(alpha=0.4),
    iaa.AveragePooling(2)
])

time_start = time.time()
batches_aug = list(aug.augment_batches(batches, background=True))  # background=True for multicore aug
time_end = time.time()

print("Augmentation done in %.2fs" % (time_end - time_start,))

for i in batches_aug:
    for j in i.images_aug:
        _id = uuid.uuid4().__str__()
        Image.fromarray(j).save(parent_path + address + _id + ".png", format="png")
