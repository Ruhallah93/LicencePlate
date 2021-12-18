import os


def get_number_analytics(directory):
    images = sorted(os.listdir(os.path.join(directory)))  # os.listdir gives a list of all files name in this path

    numbers = [0] * 10  # Counts the number of repetitions of each digit in all
    number_in_each_plate = [0] * 10  # Counts the number of repetitions of each digit in every plate
    alphabet = {"ALEF": 0, "BE": 0, "PE": 0, "TE": 0, "SE": 0, "JIM": 0, "CHE": 0, "HEY": 0, "KHE": 0, "DAL": 0,
                "ZAL": 0, "RE": 0, "ZE": 0, "ZHE": 0, "SIN": 0, "SHIN": 0, "SAD": 0, "ZAD": 0, "TA": 0, "ZA": 0,
                "EIN": 0, "GHEIN": 0, "FE": 0, "GHAF": 0, "KAF": 0, "GAF": 0, "LAM": 0, "MIM": 0, "NON": 0,
                "VAV": 0, "HE": 0, "YE": 0, "WHEEL": 0}

    for name in images:
        license_name = name.split('$')[0]
        parts = license_name.split('_')
        numbers_in_plate = parts[0] + parts[2]
        numbers_set = set(numbers_in_plate)

        # numbers
        for i in numbers_in_plate:
            numbers[int(i)] += 1

        # Alphabet
        alphabet[parts[1]] += 1

        # number in each plate
        for i in numbers_set:
            number_in_each_plate[int(i)] += 1

    return numbers, alphabet, number_in_each_plate


if __name__ == '__main__':
    # path = '../output' + '/unetData/train' + '/images'
    path = '../output/test'
    x = get_number_analytics(path)
    print(x[0])
    print(x[1])
    print(x[2])
