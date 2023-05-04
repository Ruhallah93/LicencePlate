import cv2
import random
from PIL import Image
import os
import pandas as pd
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
                   "VAV": 39, "HE": 40, "YE": 41, "WHEEL": 42, "DPLMT": 43, "SYSI": 44, "TSHFT": 45}


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
    boxes = []
    for i, glyph in enumerate(glyph_images[:-2]):
        if i == 2:
            plate.paste(glyph, (70 + w, 30), mask=glyph)
            boxes.append([70 + w, 30, glyph.size[0], glyph.size[1]])
        else:
            plate.paste(glyph, (70 + w, 25), mask=glyph)
            boxes.append([70 + w, 25, glyph.size[0], glyph.size[1]])
        w += glyph.size[0] + 3

    # last two digits
    w = 0
    for i, glyph in enumerate(glyph_images[-2:]):
        width, height = glyph.size[0], glyph.size[1]
        resized_glyph = glyph.resize((int(width * 0.75), int(height * 0.75)))
        plate.paste(resized_glyph, (485 + w, 50), mask=resized_glyph)
        boxes.append([485 + w, 50, resized_glyph.size[0], resized_glyph.size[1]])
        w += glyph.size[0] - 10
    return plate, boxes


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
    # Generate randoms Parameters
    # Set Noise Parameters

    par_state = 'rand'

    # Random
    blur_kernel_size = {'rand': random.choice(np.arange(3, 16, 2)), 'min': 3, 'max': 15}
    blur_sigma = {'rand': random.randint(3, 8), 'min': 3, 'max': 7}
    light_param = {'rand': random.randint(-170, 170), 'min': -170, 'max': 169}
    random_rect_start = {'rand': [random.randint(0, img_shape[0] - 5), random.randint(0, img_shape[1] - 5)],
                         'min': [0, 0], 'max': [img_shape[0] - 6, img_shape[1] - 6]}
    random_rect_end = {'rand': [random.randint(random_rect_start[par_state][0], img_shape[0]),
                                random.randint(random_rect_start[par_state][1], img_shape[1])],
                       'min': [random_rect_start[par_state][0], random_rect_start[par_state][1]],
                       'max': [img_shape[0] - 1, img_shape[1] - 1]}
    area = {'rand': random.choice([-1, [random_rect_start[par_state], random_rect_end[par_state]]]),
            'min': [random_rect_start[par_state], random_rect_end[par_state]], 'max': -1}
    r_circle = {'rand': random.randint(15, 31), 'min': 15, 'max': 30}
    n_circle = {'rand': 1, 'min': 1, 'max': 1}
    kernel_sigma = {'rand': random.random(), 'min': 0, 'max': 0.99}
    # r_random = random.randint(0, 4)
    r_random = {'rand': 1, 'min': 1, 'max': 1}

    min_salt = 0.15
    max_salt = 0.4
    # max_salt / (min_salt + 1) = 0.3/1.15 = 30/115
    amount_sp = {'rand': (random.random() * (max_salt - min_salt)) + min_salt, 'min': min_salt, 'max': max_salt}
    bw_random = bool(random.getrandbits(1))
    pepper_color = {'rand': random.randint(0, 128), 'min': 0, 'max': 127}
    salt_color = {'rand': random.randint(128, 256), 'min': 128, 'max': 255}

    ##################################

    # area: -1 in part 2 means go to end, Example: area=[[0, 0], [-1, 40]]
    light_noise = LightNoise(light_param=light_param[par_state], area=area[par_state])
    gradient_light_noise = GradientLightNoise(max_light_param=light_param[par_state], area=area[par_state],
                                              r=r_random[par_state])
    circular_light_noise = CircularLightNoise(light_param=light_param[par_state], n_circle=n_circle[par_state],
                                              r_circle=r_circle[par_state], kernel_sigma=kernel_sigma[par_state])
    s_p_noise = SPNoise(s_vs_p=0.5, amount=amount_sp[par_state], bw=bw_random, pepper_color=pepper_color[par_state],
                        salt_color=salt_color[par_state])
    negative_noise = NegativeNoise(blur_kernel_size=0, negative_param=-10, area=area[par_state])

    # lightNoise2 = ...
    noise_set2 = [image_noise7, image_noise8, image_noise10, light_noise,

                  gradient_light_noise, circular_light_noise, s_p_noise, negative_noise]

    # BlurNoises
    blur_noise = BlurNoise(blur_type='gaussian', blur_kernel_size=blur_kernel_size[par_state],
                           blur_sigma=blur_sigma[par_state])

    # BlurNoises array
    noise_set3 = [blur_noise]

    return noise_set1, noise_set2, noise_set3


def create_noise_sequence(img_shape, noise_dic):
    noises = []
    for i in range(1, 11):
        if noise_dic['image_' + str(i)] > 0.05:
            image_noise = ImageNoise(parent_path + f'files/noises/noise{i}.png', plate_size=img_shape)
            noises.append(image_noise)

    ##################################
    # Generate randoms Parameters
    # Set Noise Parameters

    par_state = 'rand'

    # Random
    blur_kernel_size = {'rand': random.choice(np.arange(3, 16, 2)), 'min': 3, 'max': 15}
    blur_sigma = {'rand': random.randint(3, 8), 'min': 3, 'max': 7}
    light_param = {'rand': random.randint(-170, 170), 'min': -170, 'max': 169}
    random_rect_start = {'rand': [random.randint(0, img_shape[0] - 5), random.randint(0, img_shape[1] - 5)],
                         'min': [0, 0], 'max': [img_shape[0] - 6, img_shape[1] - 6]}
    random_rect_end = {'rand': [random.randint(random_rect_start[par_state][0], img_shape[0]),
                                random.randint(random_rect_start[par_state][1], img_shape[1])],
                       'min': [random_rect_start[par_state][0], random_rect_start[par_state][1]],
                       'max': [img_shape[0] - 1, img_shape[1] - 1]}
    area = {'rand': random.choice([-1, [random_rect_start[par_state], random_rect_end[par_state]]]),
            'min': [random_rect_start[par_state], random_rect_end[par_state]], 'max': -1}
    r_circle = {'rand': random.randint(15, 31), 'min': 15, 'max': 30}
    n_circle = {'rand': 1, 'min': 1, 'max': 1}
    kernel_sigma = {'rand': random.random(), 'min': 0, 'max': 0.99}
    # r_random = random.randint(0, 4)
    r_random = {'rand': 1, 'min': 1, 'max': 1}

    min_salt = 0.15
    max_salt = 0.4
    # max_salt / (min_salt + 1) = 0.3/1.15 = 30/115
    amount_sp = {'rand': (random.random() * (max_salt - min_salt)) + min_salt, 'min': min_salt, 'max': max_salt}
    bw_random = bool(random.getrandbits(1))
    pepper_color = {'rand': random.randint(0, 128), 'min': 0, 'max': 127}
    salt_color = {'rand': random.randint(128, 256), 'min': 128, 'max': 255}

    ##################################
    if noise_dic['LightNoise'] > 0.5:
        light_noise = LightNoise(light_param=noise_dic['LightNoise_light_param'], area=area[par_state])
        noises.append(light_noise)
    if noise_dic['GradientLightNoise'] > 0.5:
        gradient_light_noise = GradientLightNoise(max_light_param=noise_dic['GradientLightNoise_light_param'],
                                                  area=area[par_state],
                                                  r=r_random[par_state])
        noises.append(gradient_light_noise)
    if noise_dic['CircularLightNoise'] > 0.5:
        circular_light_noise = CircularLightNoise(light_param=noise_dic['CircularLightNoise_light_param'],
                                                  n_circle=n_circle[par_state],
                                                  r_circle=noise_dic['CircularLightNoise_r_circle'],
                                                  kernel_sigma=noise_dic['CircularLightNoise_kernel_sigma'])
        noises.append(circular_light_noise)
    if noise_dic['SPNoise'] > 0.5:
        s_p_noise = SPNoise(s_vs_p=0.5,
                            amount=noise_dic['SPNoise_amount_sp'],
                            bw=bool(noise_dic['SPNoise_bw']),
                            pepper_color=noise_dic['SPNoise_pepper_color'],
                            salt_color=noise_dic['SPNoise_salt_color'])
        noises.append(s_p_noise)
    if noise_dic['NegativeNoise'] > 0.5:
        negative_noise = NegativeNoise(blur_kernel_size=0, negative_param=-10, area=area[par_state])
        noises.append(negative_noise)
    if noise_dic['BlurNoise'] > 0.5:
        blur_noise = BlurNoise(blur_type='gaussian',
                               blur_kernel_size=noise_dic['BlurNoise_blur_kernel_size'],
                               blur_sigma=noise_dic['BlurNoise_blur_sigma'])
        noises.append(blur_noise)

    return noises


def create_noise_dictionary(ranges_address):
    ranges = pd.read_csv(ranges_address)
    ranges.set_index("Name", inplace=True)
    noise_dic = {}

    r = random.randint(0, 4)
    functions = ['LightNoise', 'GradientLightNoise', 'CircularLightNoise', 'SPNoise', 'NegativeNoise',
                 'BlurNoise']
    for i in range(1, 11):
        functions.append('image_' + str(i))

    selected_noises = random.choices(functions, k=r)

    for i in range(1, 11):
        if 'image_' + str(i) in selected_noises:
            noise_dic['image_' + str(i)] = 1
        else:
            noise_dic['image_' + str(i)] = 0

    if 'LightNoise' in selected_noises:
        noise_dic['LightNoise'] = 1
        noise_dic['LightNoise_light_param'] = random.randint(float(ranges.loc["LightNoise_light_param_min"]),
                                                             float(ranges.loc["LightNoise_light_param_max"]))
    else:
        noise_dic['LightNoise'] = 0
        noise_dic['LightNoise_light_param'] = 0

    if 'GradientLightNoise' in selected_noises:
        noise_dic['GradientLightNoise'] = 1
        noise_dic['GradientLightNoise_light_param'] = random.randint(
            float(ranges.loc["GradientLightNoise_light_param_min"]),
            float(ranges.loc["GradientLightNoise_light_param_max"]))
    else:
        noise_dic['GradientLightNoise'] = 0
        noise_dic['GradientLightNoise_light_param'] = 0

    if 'CircularLightNoise' in selected_noises:
        noise_dic['CircularLightNoise'] = 1
        noise_dic['CircularLightNoise_light_param'] = random.randint(
            float(ranges.loc["CircularLightNoise_light_param_min"]),
            float(ranges.loc["CircularLightNoise_light_param_max"]))
        noise_dic['CircularLightNoise_r_circle'] = random.randint(float(ranges.loc["CircularLightNoise_r_circle_min"]),
                                                                  float(ranges.loc["CircularLightNoise_r_circle_max"]))
        noise_dic['CircularLightNoise_kernel_sigma'] = random.random()
    else:
        noise_dic['CircularLightNoise'] = 0
        noise_dic['CircularLightNoise_light_param'] = 0
        noise_dic['CircularLightNoise_r_circle'] = 0
        noise_dic['CircularLightNoise_kernel_sigma'] = 0

    if 'SPNoise' in selected_noises:
        noise_dic['SPNoise'] = 1
        min_salt = float(ranges.loc["SPNoise_amount_sp_min"])
        max_salt = float(ranges.loc["SPNoise_amount_sp_max"])
        noise_dic['SPNoise_amount_sp'] = (random.random() * (max_salt - min_salt)) + min_salt
        noise_dic['SPNoise_bw'] = random.getrandbits(int(ranges.loc["SPNoise_bw_max"]))
        noise_dic['SPNoise_pepper_color'] = random.randint(float(ranges.loc["SPNoise_pepper_color_min"]),
                                                           float(ranges.loc["SPNoise_pepper_color_max"]))
        noise_dic['SPNoise_salt_color'] = random.randint(float(ranges.loc["SPNoise_salt_color_min"]),
                                                         float(ranges.loc["SPNoise_salt_color_max"]))
    else:
        noise_dic['SPNoise'] = 0
        noise_dic['SPNoise_amount_sp'] = 0
        noise_dic['SPNoise_bw'] = 0
        noise_dic['SPNoise_pepper_color'] = 0
        noise_dic['SPNoise_salt_color'] = 0

    if 'NegativeNoise' in selected_noises:
        noise_dic['NegativeNoise'] = 1
    else:
        noise_dic['NegativeNoise'] = 0

    if 'BlurNoise' in selected_noises:
        noise_dic['BlurNoise'] = 1
        noise_dic['BlurNoise_blur_sigma'] = random.randint(float(ranges.loc["BlurNoise_blur_sigma_min"]),
                                                           float(ranges.loc["BlurNoise_blur_sigma_max"]))
        noise_dic['BlurNoise_blur_kernel_size'] = random.choice(
            np.arange(float(ranges.loc["BlurNoise_blur_kernel_size_min"]),
                      float(ranges.loc["BlurNoise_blur_kernel_size_max"]), 2))
    else:
        noise_dic['BlurNoise'] = 0
        noise_dic['BlurNoise_blur_sigma'] = 0
        noise_dic['BlurNoise_blur_kernel_size'] = 0

    r = random.randint(0, 3)
    selected_directions = random.choices(['pitch', 'yaw', 'roll'], k=r)
    if 'pitch' in selected_directions:
        noise_dic['pitch'] = np.random.randint(float(ranges.loc["pitch_min"]), float(ranges.loc["pitch_max"]))
    else:
        noise_dic['pitch'] = 0
    if 'yaw' in selected_directions:
        noise_dic['yaw'] = np.random.randint(float(ranges.loc["yaw_min"]), float(ranges.loc["yaw_max"]))
    else:
        noise_dic['yaw'] = 0
    if 'roll' in selected_directions:
        noise_dic['roll'] = np.random.randint(float(ranges.loc["roll_min"]), float(ranges.loc["roll_max"]))
    else:
        noise_dic['roll'] = 0

    noise_dic['scale'] = random.uniform(float(ranges.loc["scale_min"]), float(ranges.loc["scale_max"]))
    return noise_dic


# Set background for license plate
def set_background(img, mask, merged_boxes,
                   background_size,
                   paste_point: tuple = (0, 0),
                   random_scale=0.5,
                   background_path="r"):
    """
    img: plate image
    width, height: width & height of background, if width == -1: make a random number for width
    and if width < def license plate size: width = lp_size[0]
    random_scale: =0, dont scale img. !=0, resize img with random scale between img.dim -+ \
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
    # Make (random) scale and then set new_width and new_height
    new_height = int(background_size[0] * random_scale)
    new_width = int(background_size[1] * random_scale)
    background = background.resize((new_height, new_width), Image.ANTIALIAS)
    mask_background = mask_background.resize((new_height, new_width), Image.ANTIALIAS)

    new_merged_boxes[:, 0] = (new_merged_boxes[:, 0] / background_size[1]) * new_width
    new_merged_boxes[:, 1] = (new_merged_boxes[:, 1] / background_size[0]) * new_height
    new_merged_boxes[:, 2] = (new_merged_boxes[:, 2] / background_size[1]) * new_width
    new_merged_boxes[:, 3] = (new_merged_boxes[:, 3] / background_size[0]) * new_height

    background = np.array(background)

    # visualization(background, [mask_background], boxes=new_merged_boxes)

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


def rgba_2_bgr(img):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    if img.shape[2] == 4:
        street = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return np.array(Image.fromarray(street))
    else:
        return np.array(img)


def rbga2gray(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY), cv2.COLOR_GRAY2BGR)


def visualization(main_image, images=None, boxes=None, waitKey=0):
    main_image = rgba_2_bgr(main_image)
    if boxes is not None:
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(main_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # cv2.line(main_image, (0, 100), (200, 0), (0, 0, 255), 2)
    # cv2.line(main_image, (200, 0), (400, 100), (0, 0, 255), 2)

    cv2.imshow('main', main_image)
    if images is not None:
        for i, img in enumerate(images):
            cv2.imshow(str(i), rgba_2_bgr(img))
    return cv2.waitKey(waitKey)


def resize(img, sizes):
    if img.size[0] > sizes[0]:
        img = img.resize((sizes[0], int(sizes[0] * img.size[1] / img.size[0])))
    if img.size[1] > sizes[1]:
        img = img.resize((int(sizes[1] * img.size[1] / img.size[1]), sizes[0]))
    w_background = Image.new('RGB', sizes, (255, 255, 255))
    w_background.paste(img, (0, 0))
    return w_background


def resize2(img, sizes):
    img = img.resize(sizes)
    w_background = Image.new('RGB', sizes, (255, 255, 255))
    w_background.paste(img, (0, 0))
    visualization(w_background)
    return w_background
