"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.

Usage:
       python capsulenet-multi-gpu.py
       python capsulenet-multi-gpu.py --gpus 2
       ... ...

Result:
    About 55 seconds per epoch on two GTX1080Ti GPU cards

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

from keras import optimizers
from keras import backend as K

K.set_image_data_format('channels_last')

from capsulenet import CapsNet, margin_loss, manipulate_latent, test


def count_files(address):
    total = 0
    for root, dirs, files in os.walk(address):
        total += len(files)
    return total


def train(model, train_directory, valid_directory, image_size, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights.h5', monitor='val_capsnet_loss',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon])

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(directory, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)
        generator = train_datagen.flow_from_directory(directory, batch_size=batch_size, target_size=image_size)
        while 1:
            x_batch, y_batch = generator.next()
            x_batch /= 255
            yield ([x_batch, y_batch], [y_batch, x_batch])

    def valid_generator(directory, batch_size):
        train_datagen = ImageDataGenerator()
        generator = train_datagen.flow_from_directory(directory, batch_size=batch_size, target_size=image_size)
        while 1:
            x_batch, y_batch = generator.next()
            x_batch /= 255
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit(generator=train_generator(train_directory, args.batch_size, args.shift_fraction),
              steps_per_epoch=int(count_files(train_directory) / args.batch_size),
              epochs=args.epochs,
              validation_data=valid_generator(valid_directory, args.batch_size),
              validation_steps=int(count_files(valid_directory) / args.batch_size),
              callbacks=[log, tb, checkpoint, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


if __name__ == "__main__":
    import numpy as np
    import tensorflow as tf
    import os
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks
    from keras.utils.vis_utils import plot_model
    from keras.utils import multi_gpu_model

    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=300, type=int)
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', default=0, type=int,
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--gpus', default=2, type=int)

    parser.add_argument('--train_address', type=str, default='synthetic_plates/output/yolo/train/glyphs',
                        help='The address of train dataset')
    parser.add_argument('--valid_address', type=str, default='synthetic_plates/output/yolo/valid/glyphs',
                        help='The address of valid dataset')
    parser.add_argument('--test_address', type=str, default='synthetic_plates/output/yolo/test/glyphs',
                        help='The address of test dataset')
    parser.add_argument('--glyph_size', nargs='+', type=int, default=[80, 80, 3], help='size of saved glyphs')
    parser.add_argument('--n_class', type=int, default=27, help='number of threads to run')
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define model
    with tf.device('/cpu:0'):
        model, eval_model, manipulate_model = CapsNet(input_shape=args.glyph_size,
                                                      n_class=args.n_class,
                                                      routings=args.routings,
                                                      batch_size=args.batch_size)
    model.summary()
    plot_model(model, to_file=args.save_dir + '/model.png', show_shapes=True)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        # define muti-gpu model
        multi_model = multi_gpu_model(model, gpus=args.gpus)
        train(model=multi_model,
              train_directory=args.train_address,
              valid_directory=args.valid_address,
              image_size=args.glyph_size[:-1],
              args=args)
        model.save_weights(args.save_dir + '/trained_model.h5')
        print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
        test(model=eval_model, test_directory=args.test_address, image_size=args.glyph_size[:-1], args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        # manipulate_latent(manipulate_model, args.test_address, image_size=args.glyph_size[:-1], args=args)
        test(model=eval_model, test_directory=args.test_address, image_size=args.glyph_size[:-1], args=args)
