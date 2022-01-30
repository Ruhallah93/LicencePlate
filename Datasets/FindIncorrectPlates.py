import glob
import os.path

file_list = []
for path in glob.glob("/home/ruhiii/Downloads/Telegram Desktop/Ruhallah/*.txt"):
    with open(path, 'r') as file1:
        with open("/home/ruhiii/Downloads/Telegram Desktop/OriginDataSet/Ruhallah/" + os.path.basename(path),
                  'r') as file2:
            difference = set(file1).difference(file2)
            if len(difference) > 0:
                print(os.path.basename(path).split('.')[0])
                file_list.append(os.path.basename(path))
print("Total files: ", len(file_list))
