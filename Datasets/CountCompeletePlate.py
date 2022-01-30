import glob
import json
import os.path
from collections import Counter
from PIL import Image

from synthetic_plates.utils.Utils import visualization

dataset_letter = ["b", "d", "ein", "q", "h", "j", "l", "m",
                  "n", "RE", "sad", "s", "ta", "t", "v", "malol", "y"]
support_letter = ["BE", "DAL", "EIN", "GHAF", "HE", "JIM", "LAM", "MIM",
                  "NON", "RE", "SAD", "SIN", "TA", "TE", "VAV", "WHEEL", "YE"]
counter = 0
counter_unsupported_labels = 0
numbers = []
a = []
for path in glob.glob("/home/ruhiii/Downloads/CharFinder/labels/*.json"):
    chars = json.load(open(path))
    chars = sorted(chars, key=lambda x: x['x'])
    if len(chars) == 8:
        name_of_file = "{}_{}_{}".format(chars[0]["char_en"] + chars[1]["char_en"],
                                         chars[2]["char_en"].upper(),
                                         chars[3]["char_en"] + chars[4]["char_en"] + chars[5]["char_en"] + chars[6][
                                             "char_en"] + chars[7]["char_en"])
        numbers.append(name_of_file)
        if chars[2]["char_en"] not in dataset_letter:
            counter_unsupported_labels += 1
            a.append(chars[2]["char_en"])
            # print(chars[2]["char_en"])
            # img = Image.open(f"/home/ruhiii/Downloads/CharFinder/images/{os.path.basename(path).split('.')[0]}.jpg")
            # boxes = [[x, y, w, h] for h, w, x, y in [list(char.values())[-4:] for char in chars]]
            # visualization(main_image=img, boxes=boxes)
        counter += 1
# print({k: v for k, v in sorted(all_dataset_letter_dic.items(), key=lambda item: item[1])})
print("Total Plates: ", len(glob.glob("/home/ruhiii/Downloads/CharFinder/labels/*.json")))
print("Incorrect Plates", len(glob.glob("/home/ruhiii/Downloads/CharFinder/labels/*.json")) - counter)
print("Correct Plates: ", counter)
print("Unsupported Plates: ", counter_unsupported_labels)
print(Counter(a))
duplicates = dict((k, v) for k, v in Counter(numbers).items() if v > 1)
print("Duplicates in Correct Plates", len(duplicates))
print(duplicates)
print("Used Plates: ", counter - counter_unsupported_labels - len(duplicates))
