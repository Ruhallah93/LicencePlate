import cv2
import random
from PIL import Image
import os
from .noise.BlurNoise import BlurNoise
from .noise.CircularLightNoise import CircularLightNoise
from .noise.GradientLightNoise import GradientLightNoise
from .noise.NegativeNoise import NegativeNoise
from .noise.ImageNoise import ImageNoise
from .noise.LightNoise import LightNoise
from .noise.SPNoise import SPNoise
import PIL.ImageOps
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch
import glob
import uuid

separetor = os.sep
package_directory = os.path.dirname(os.path.abspath(__file__))
parent_path = package_directory[:package_directory.rindex(separetor)] + separetor

# Characters of Letters and Numbers in Plates
numbers = [str(i) for i in range(0, 10)]

# ignored: "FE", "ZE", "SHIN", "PE", "SE", "ALEF","TASHRIFAT","KAF","GAF"
letters = ["BE", "TE", "JIM", "DAL", "RE", "SIN", "SAD", "TA", "EIN", "GHAF", "LAM", "MIM", "NON", "VAV", "HE",
           "YE", "WHEEL"]
letter_to_class = {"ALEF": 10, "BE": 11, "PE": 12, "TE": 13, "SE": 14, "JIM": 15, "CHE": 16, "HEY": 17, "KHE": 18,
                   "DAL": 19, "ZAL": 20, "RE": 21, "ZE": 22, "ZHE": 23,
                   "SIN": 24, "SHIN": 25, "SAD": 26, "ZAD": 27, "TA": 28, "ZA": 29, "EIN": 30, "GHEIN": 31, "FE": 32,
                   "GHAF": 33, "KAF": 34, "GAF": 35, "LAM": 36, "MIM": 37, "NON": 38,
                   "VAV": 39, "HE": 40, "YE": 41, "WHEEL": 42}


def get_new_plate_number():
    return [random.choice(numbers),
            random.choice(numbers),
            random.choice(letters),
            random.choice(numbers),
            random.choice(numbers),
            random.choice(numbers),
            random.choice(numbers),
            random.choice(numbers)]


# Returns Address of a glyph image given font, and glyph name
def get_glyph_address(glyph_name, c='grayscale'):
    if c == 'colorful':
        return os.path.join(parent_path + "files/Glyphs/b_roya_color", "{}.png".format(glyph_name))
    else:
        return os.path.join(parent_path + "files/Glyphs/b_roya", "{}.png".format(glyph_name))


def get_template(plate):
    temp_address = os.path.join(*(parent_path, 'files', 'templates')) + separetor
    if plate[2] in ["D", "S"]:
        background = Image.open(temp_address + "template-diplomat.png").convert("RGBA")
    elif plate[2] in ["TE", "EIN"]:
        background = Image.open(temp_address + "template-ommomi.png").convert("RGBA")
    elif plate[2] in ["FE", "ZE"]:
        background = Image.open(temp_address + "template-defa.png").convert("RGBA")
    elif plate[2] in ["SHIN"]:
        background = Image.open(temp_address + "template-artesh.png").convert("RGBA")
    elif plate[2] in ["PE"]:
        background = Image.open(temp_address + "template-police.png").convert("RGBA")
    elif plate[2] in ["SE"]:
        background = Image.open(temp_address + "template-sepah.png").convert("RGBA")
    elif plate[2] in ["ALEF"]:
        background = Image.open(temp_address + "template-dolati.png").convert("RGBA")
    elif plate[2] in ["TASHRIFAT"]:
        background = Image.open(temp_address + "template-tashrifat.png").convert("RGBA")
    elif plate[2] in ["KAF"]:
        background = Image.open(temp_address + "template-.png").convert("RGBA")
    elif plate[2] in ["GAF"]:
        background = Image.open(temp_address + "template-gozar.png").convert("RGBA")
    else:
        background = Image.open(temp_address + "template-base.png").convert("RGBA")
    return background


def apply_glyphs(plate, mask_state):
    def invert(img):
        r, g, b, a = img.split()
        rgb_image = Image.merge('RGB', (r, g, b))
        inverted_image = PIL.ImageOps.invert(rgb_image)
        r2, g2, b2 = inverted_image.split()
        final_transparent_image = Image.merge('RGBA', (r2, g2, b2, a))
        return final_transparent_image

    def border(img):
        r, g, b, a = img.split()
        A = np.array(a)
        A[:, [0, -1]] = 0
        rgb_image = Image.merge('RGB', (r, g, b))
        return Image.merge('RGBA', (r, g, b, PIL.Image.fromarray(A)))

    # Get Glyph images of plate characters
    if plate[2] in ["FE", "ZE", "PE", "SE", "ALEF", "TASHRIFAT"]:
        glyph_images = [invert(Image.open(get_glyph_address(glyph, c='grayscale')).convert("RGBA")) for glyph in plate]
        glyph_images_mask = [border(Image.open(get_glyph_address(glyph, mask_state)).convert("RGBA")) for glyph in
                             plate]
    else:
        glyph_images = [Image.open(get_glyph_address(glyph, c='grayscale')).convert("RGBA") for glyph in plate]
        glyph_images_mask = [Image.open(get_glyph_address(glyph, mask_state)).convert("RGBA") for glyph in plate]

    return glyph_images, glyph_images_mask


def adjust_glyphs(glyph_images, plate):
    w = 0
    for i, glyph in enumerate(glyph_images[:-2]):
        if i == 2:
            plate.paste(glyph, (70 + w, 30), mask=glyph)
        else:
            plate.paste(glyph, (70 + w, 25), mask=glyph)
        w += glyph.size[0] + 3

    # last two digits
    w = 0
    for i, glyph in enumerate(glyph_images[-2:]):
        width, height = glyph.size[0], glyph.size[1]
        resized_glyph = glyph.resize((int(width * 0.75), int(height * 0.75)))
        plate.paste(resized_glyph, (485 + w, 50), mask=resized_glyph)
        w += glyph.size[0] - 10
    return plate


def create_noise_palettes(img_shape):
    image_noise1 = ImageNoise(parent_path + 'files/noises/noise1.png', plate_size=img_shape)
    image_noise2 = ImageNoise(parent_path + 'files/noises/noise2.png', plate_size=img_shape)
    image_noise3 = ImageNoise(parent_path + 'files/noises/noise3.png', plate_size=img_shape)
    image_noise4 = ImageNoise(parent_path + 'files/noises/noise4.png', plate_size=img_shape)
    image_noise5 = ImageNoise(parent_path + 'files/noises/noise5.png', plate_size=img_shape)
    image_noise6 = ImageNoise(parent_path + 'files/noises/noise6.png', plate_size=img_shape)
    image_noise9 = ImageNoise(parent_path + 'files/noises/noise9.png', plate_size=img_shape)
    noise_set1 = [image_noise1, image_noise2, image_noise3, image_noise4, image_noise5, image_noise6, image_noise9]

    image_noise7 = ImageNoise(parent_path + 'files/noises/noise7.png', plate_size=img_shape)
    image_noise8 = ImageNoise(parent_path + 'files/noises/noise8.png', plate_size=img_shape)
    image_noise10 = ImageNoise(parent_path + 'files/noises/noise10.png', plate_size=img_shape)

    ##################################
    # Generate randoms
    blur_kernel_size = random.choice(np.arange(3, 8, 2))
    blur_sigma = random.randint(3, 8)
    light_param = random.randint(-170, 170)
    random_rect_start = [random.randint(0, img_shape[0] - 5), random.randint(0, img_shape[1] - 5)]
    random_rect_end = [random.randint(random_rect_start[0], img_shape[0]),
                       random.randint(random_rect_start[1], img_shape[1])]
    area = random.choice([-1, [random_rect_start, random_rect_end]])
    r_circle = random.randint(15, 31)
    n_circle = random.randint(1, 3)
    kernel_sigma = random.random()
    # r_random = random.randint(0, 4)
    r_random = 1

    min_salt = 0.15
    max_salt = 0.3
    # max_salt / (min_salt + 1) = 0.3/1.15 = 30/115
    amount_sp = (random.random() * (max_salt - min_salt)) + min_salt
    bw_random = bool(random.getrandbits(1))
    pepper_color = random.randint(0, 128)
    salt_color = random.randint(128, 256)

    ##################################

    # area: -1 in part 2 means go to end, Example: area=[[0, 0], [-1, 40]]
    light_noise = LightNoise(light_param=light_param, area=area)
    gradient_light_noise = GradientLightNoise(max_light_param=light_param, area=area, r=r_random)
    circular_light_noise = CircularLightNoise(light_param=light_param, n_circle=n_circle,
                                              r_circle=r_circle, kernel_sigma=kernel_sigma)
    s_p_noise = SPNoise(s_vs_p=0.5, amount=amount_sp, bw=bw_random, pepper_color=pepper_color, salt_color=salt_color)
    negative_noise = NegativeNoise(blur_kernel_size=0, negative_param=-10, area=area)

    # lightNoise2 = ...
    noise_set2 = [image_noise7, image_noise8, image_noise10, light_noise,
                  gradient_light_noise, circular_light_noise, s_p_noise, negative_noise]

    # BlurNoises
    blur_noise = BlurNoise(blur_type='gaussian', blur_kernel_size=blur_kernel_size, blur_sigma=blur_sigma)

    # BlurNoises array
    noise_set3 = [blur_noise]

    return noise_set1, noise_set2, noise_set3


# Set background for license plate
def set_background(img, mask, merged_boxes,
                   background_size,
                   paste_point: tuple = (0, 0),
                   min_random_scale=0.5,
                   background_path="r"):
    """
    img: plate image
    width, height: width & height of background, if width == -1: make a random number for width
    and if width < def license plate size: width = lp_size[0]
    random_scale: =0, dont scale img. !=0, resize img with random scale between img.dim -+ img.dim * random_scale
    background_path: background path image, if pass this argument r, set a random static color for background
    position: position of license plate in the background, if pass -1 generate a random position
    """

    new_merged_boxes = np.array(merged_boxes)

    # Add background to plate
    if background_path == "r":
        r, g, b = random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)
        background = Image.new('RGBA', background_size, (r, g, b, 255))
    else:
        background = Image.open(background_path).convert("RGBA").resize(background_size)
    mask_background = Image.new('RGBA', background_size, (255, 255, 255, 255))

    y = paste_point[0]
    x = paste_point[1]

    background.paste(Image.fromarray(img), (x, y))
    mask_background.paste(Image.fromarray(mask), (x, y))

    new_merged_boxes[:, 0] = x + new_merged_boxes[:, 0]
    new_merged_boxes[:, 1] = y + new_merged_boxes[:, 1]

    # Resize plate
    if min_random_scale != 0:
        # Make (random) scale and then set new_width and new_height
        random_scale = random.uniform(min_random_scale, 1)
        new_height = int(background_size[0] * random_scale)
        new_width = int(background_size[1] * random_scale)
        background = background.resize((new_height, new_width), Image.ANTIALIAS)
        mask_background = mask_background.resize((new_height, new_width), Image.ANTIALIAS)

        new_merged_boxes[:, 0] = (new_merged_boxes[:, 0] / background_size[1]) * new_width
        new_merged_boxes[:, 1] = (new_merged_boxes[:, 1] / background_size[0]) * new_height
        new_merged_boxes[:, 2] = (new_merged_boxes[:, 2] / background_size[1]) * new_width
        new_merged_boxes[:, 3] = (new_merged_boxes[:, 3] / background_size[0]) * new_height

    background = np.array(background)

    # for box in new_merged_boxes:
    #     x, y, w, h = box
    #     cv2.rectangle(background, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # visual = cv2.cvtColor(background, cv2.COLOR_BGRA2RGB)
    # visual = Image.fromarray(visual)
    # visual = np.array(visual)
    # cv2.imshow('background', visual)
    # cv2.imshow('mask_background', np.array(mask_background))
    # cv2.waitKey()

    return np.array(background), np.array(mask_background), new_merged_boxes


def crop(img, mask, plate_box, glyph_boxes):
    nx, ny, nw, nh = plate_box
    img = img[ny:ny + nh, nx:nx + nw]
    mask = mask[ny:ny + nh, nx:nx + nw]
    new_glyph_boxes = []
    for x, y, w, h in glyph_boxes:
        new_glyph_boxes.append((x - nx, y - ny, w, h))
    return img, mask, (plate_box[0] - nx, plate_box[1] - ny, plate_box[2], plate_box[3]), new_glyph_boxes


def augmentation(from_directory, to_directory, nb_batches=5):
    if not os.path.exists(parent_path + to_directory):
        os.makedirs(parent_path + to_directory)

    images = []

    types = ('*.png', '*.jpg', '*.jpeg')
    for type_ in types:
        for filename in glob.glob(parent_path + from_directory + '/' + type_):
            im = Image.open(filename)
            images.append(np.array(im))

    batches = [UnnormalizedBatch(images=images) for _ in range(nb_batches)]

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

    batches_aug = list(aug.augment_batches(batches, background=True))  # background=True for multicore aug

    for i in batches_aug:
        for j in i.images_aug:
            _id = uuid.uuid4().__str__()
            Image.fromarray(j).save(parent_path + to_directory + _id + ".png", format="png")
