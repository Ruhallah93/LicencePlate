import argparse
from utils.Utils import augmentation
import os

if __name__ == '__main__':
    # For test: set workers default to 1
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_directory', type=str, default='files/streets')
    parser.add_argument('--to_directory', type=str, default='files/aug/streets')
    parser.add_argument('--agn_nb_batches', type=int, default=2, help='generate nb_batches * dataset')
    opt = parser.parse_args()

    opt.to_directory = opt.to_directory + os.sep if opt.to_directory[-1] != os.sep else opt.to_directory

    augmentation(opt.from_directory, opt.to_directory, nb_batches=opt.agn_nb_batches)
