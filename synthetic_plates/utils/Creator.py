from .Utils import *
from .Perspective import create_perspective
from PIL import Image
import random

import os

separetor = os.sep
package_directory = os.path.dirname(os.path.abspath(__file__))
parent_path = package_directory[:package_directory.rindex(separetor)] + separetor


def get_new_plate(img_size, mask_state='grayscale'):
    """
        mask_state: grayscale or colorful
    """
    plate = get_new_plate_number()

    # Create a blank image with size of templates
    # and add the background and glyph images
    new_plate = Image.new('RGBA', (600, 132), (0, 0, 0, 0))
    mask = new_plate.copy()

    # Get the background associated with the letter plate[2]
    background = get_template(plate)

    # Merge the plate with the background
    new_plate.paste(background, (0, 0))
    white_background = Image.open(parent_path + "files/templates/white.png").convert("RGBA")
    mask.paste(white_background, (0, 0))

    # Get the glyphs of plate
    glyph_images, glyph_images_mask = apply_glyphs(plate, mask_state)

    # adding glyph images with 11 pixel margin
    new_plate = adjust_glyphs(glyph_images, new_plate)
    mask = adjust_glyphs(glyph_images_mask, mask)

    _newPlate = new_plate.resize((312, 70), Image.ANTIALIAS)
    mask = mask.resize((312, 70), Image.ANTIALIAS)

    # Add noise to plate
    noise_set1, noise_set2, noise_set3 = create_noise_palettes()
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
    perspective_plate, mask, bounding_boxes = create_perspective(_newPlate, mask, img_size=img_size, noises=noises,
                                                                 perspective_type=np.random.random_integers(0, 5),
                                                                 pad=(50, 50, 50, 50))

    return plate, perspective_plate, mask, bounding_boxes
