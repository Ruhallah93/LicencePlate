import pandas as pd
import argparse
import os

from utils.Utils import letters


def get_number_analytics(directory):
    images = sorted(os.listdir(os.path.join(directory)))  # os.listdir gives a list of all files name in this path

    info_overall = dict.fromkeys([str(i) for i in range(10)] + letters, 0)
    info_per_plate = dict.fromkeys([str(i) for i in range(10)] + letters, 0)

    for name in images:
        license_name = name.split('$')[0]
        parts = license_name.split('_')
        numbers_in_plate = parts[0] + parts[2]
        numbers_set = set(numbers_in_plate)

        # numbers
        for i in numbers_in_plate:
            info_overall[str(i)] += 1
        for i in numbers_set:
            info_per_plate[str(i)] += 1

        # Alphabet
        info_overall[parts[1]] += 1
        info_per_plate[parts[1]] += 1

    info = pd.DataFrame.from_dict({"Overall": info_overall, "Per Plate": info_per_plate})
    info['Ratio (O/P)'] = info[info.columns[0]] / info[info.columns[1]]
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str, default='output/unet', help='Dataset path')
    parser.add_argument('--save_path', type=str, default='', help='Dataset path')
    opt = parser.parse_args()

    info = get_number_analytics(opt.address)
    info.to_csv(opt.save_path + "statistics.csv")
    print(info)
