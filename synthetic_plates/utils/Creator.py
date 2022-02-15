from .Utils import *
from .Perspective import create_perspective
from PIL import Image
import random

import os

separetor = os.sep
package_directory = os.path.dirname(os.path.abspath(__file__))
parent_path = package_directory[:package_directory.rindex(separetor)] + separetor


def get_new_plate(img_size,
                  rotation_maximums,
                  plate_size=(600, 132),
                  paste_point=(0, 0),
                  mask_state='grayscale',
                  pad: tuple = (50, 50, 50, 50),
                  random_scale=0.5,
                  background_path="r"):
    """
        img_size: size of whole image,
        rotation_maximums: maximum degrees of rotation in perspective task,
        plate_size: plate's size in the image,
        attach_point: where the plate pastes to image,
        mask_state: grayscale or colorful
        pad: padding for the image (top,bottom,left,right),
        random_scale: scale rate of image,
        background_path: the address of image's background
    """
    plate = get_new_plate_number()

    # Create a blank image with size of templates
    # and add the template and glyph images
    new_plate = Image.new('RGBA', (600, 132), (0, 0, 0, 0))
    mask = new_plate.copy()

    # Get the template associated with the letter plate[2]
    template = get_template(plate)

    # Merge the plate with the template
    new_plate.paste(template, (0, 0))
    white_template = Image.open(parent_path + "files/templates/white.png").convert("RGBA")
    mask.paste(white_template, (0, 0))

    # Get the glyphs of plate
    glyph_images, glyph_images_mask = apply_glyphs(plate, mask_state)

    # adding glyph images with 11 pixel margin
    new_plate, plate_boxes = adjust_glyphs(glyph_images, new_plate)
    mask, mask_boxes = adjust_glyphs(glyph_images_mask, mask)

    # visualization(new_plate, boxes=plate_boxes)
    # visualization(mask, boxes=mask_boxes)

    new_plate_2 = new_plate.resize(plate_size, Image.ANTIALIAS)
    mask = mask.resize(plate_size, Image.ANTIALIAS)

    # Add noise to plate
    noise_set1, noise_set2, noise_set3 = create_noise_palettes(plate_size)
    r = random.randint(0, 3)
    noises = []
    if r == 1:
        noises = [random.choice(noise_set2 + noise_set1)]
    elif r == 2:
        r_blur_list = [0] * 64 + [1] * 35 + [2] * 1
        r_blur = random.choice(r_blur_list)
        if r_blur == 0:
            noises = [random.choice(noise_set1), random.choice(noise_set2)]
        elif r_blur == 1:
            noises = [random.choice(noise_set2), random.choice(noise_set3)]
        else:
            noises = [random.choice(noise_set1), random.choice(noise_set2), random.choice(noise_set3)]

    # Make the plate perspective
    perspective_plate, mask, bounding_boxes, plate_box = create_perspective(new_plate_2, mask,
                                                                            img_size=img_size,
                                                                            plate_size=plate_size,
                                                                            paste_point=paste_point,
                                                                            noises=noises,
                                                                            perspective_type=np.random.random_integers(
                                                                                0, 5),
                                                                            rotation_maximums=rotation_maximums,
                                                                            pad=pad,
                                                                            random_scale=random_scale,
                                                                            background_path=background_path)

    return plate, perspective_plate, mask, bounding_boxes, plate_box
