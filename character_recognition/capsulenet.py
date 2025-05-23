"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import cv2
import PIL.ImageOps
import os
import argparse

tf.keras.backend.set_image_data_format('channels_last')


def rgba_2_bgr(img):
    if isinstance(img, PIL.Image.Image):
        img = np.array(img)
    if img.shape[2] == 4:
        street = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return np.array(Image.fromarray(street))
    else:
        return np.array(img)


def visualization(main_image, images=None, boxes=None, waitKey=0):
    main_image = cv2.cvtColor(np.array(main_image), cv2.COLOR_BGR2RGB)
    main_image = Image.fromarray(main_image, mode="RGB")
    main_image = rgba_2_bgr(main_image)
    if boxes is not None:
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(main_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imshow('main', main_image)
    if images is not None:
        for i, img in enumerate(images):
            cv2.imshow(str(i), rgba_2_bgr(img))
    cv2.waitKey(waitKey)


def count_files(address):
    total = 0
    for root, dirs, files in os.walk(address):
        total += len(files)
    return total


def CapsNet(input_shape, n_class, routings, batch_size):
    """
    A Capsule Network on MNIST.
    :param input_shape: no_labels shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :param batch_size: size of batch
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = tf.keras.layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu',
                                   name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = tf.keras.layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = tf.keras.models.Sequential(name='decoder')
    decoder.add(tf.keras.layers.Dense(512, activation='relu', input_dim=16 * n_class))
    decoder.add(tf.keras.layers.Dense(1024, activation='relu'))
    decoder.add(tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(tf.keras.layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = tf.keras.models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = tf.keras.models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = tf.keras.layers.Input(shape=(n_class, 16))
    noised_digitcaps = tf.keras.layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = tf.keras.models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    # return tf.reduce_mean(tf.square(y_pred))
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))


def train(model,  # type: models.Model
          train_directory, valid_directory, image_size, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param no_labels: a tuple containing training and testing no_labels, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """

    # callbacks
    log = tf.keras.callbacks.CSVLogger(args.save_dir + '/log.csv')
    # '/weights-{epoch:02d}.h5'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(args.save_dir + '/weights.h5', monitor='val_loss',
                                                    save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    # tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without no_labels augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with no_labels augmentation ---------------------------------------------------------------------#
    def train_generator(directory, batch_size, shift_fraction=0.):
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=shift_fraction,
                                                                        height_shift_range=shift_fraction)
        generator = train_datagen.flow_from_directory(directory, batch_size=batch_size, target_size=image_size)
        while 1:
            x_batch, y_batch = generator.next()
            x_batch /= 255
            yield ([x_batch, y_batch], [y_batch, x_batch])

    def valid_generator(directory, batch_size):
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        generator = train_datagen.flow_from_directory(directory, batch_size=batch_size, target_size=image_size)
        while 1:
            x_batch, y_batch = generator.next()
            x_batch /= 255
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # Training with no_labels augmentation. If shift_fraction=0., no augmentation.
    # train_generator(train_directory, args.batch_size, args.shift_fraction)
    model.fit(train_generator(train_directory, args.batch_size, args.shift_fraction),
              steps_per_epoch=int(count_files(train_directory) / args.batch_size),
              epochs=args.epochs,
              validation_data=valid_generator(valid_directory, args.batch_size),
              validation_steps=int(count_files(valid_directory) / args.batch_size),
              callbacks=[log, checkpoint, lr_decay])
    # End: Training with no_labels augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, test_directory, image_size, args):
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_directory,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=args.batch_size,
        image_size=image_size,
        shuffle=False,
    )
    images = []
    labels = []
    for image, label in test_ds.take(-1):
        # visualization(main_image=image[0])
        images.extend(image / 255)
        labels.extend(label)
    images = np.array(images)
    labels = np.array(labels)
    print(images.shape)
    print(labels.shape)

    y_pred, x_recon = model.predict(images, batch_size=args.batch_size, steps=int(images.shape[0] / args.batch_size))
    print('-' * 30 + 'Begin: test' + '-' * 30)
    y_pred = np.argmax(y_pred, 1)
    print('Test acc:', np.sum(y_pred == labels) / labels.shape[0])

    # predictions = tf.nn.sigmoid(predictions)
    # predictions = tf.where(predictions < 0.5, 0, 1)

    print(classification_report(labels, y_pred))
    print(confusion_matrix(labels, y_pred))

    diff = np.where(labels != y_pred)
    for i, (img, label) in enumerate(zip(images[diff], labels[diff])):
        img = tf.keras.preprocessing.image.array_to_img(img * 255, scale=False)
        img.save('wrong_predicts/{}${}.png'.format(label, i), format="PNG")

    # print('Test acc:', model.evaluate_generator(test_ds))
    #
    # for images, labels in test_ds.take(1):
    #     img = combine_images(np.concatenate([images[:50], x_recon[:50]]))
    # image = img * 255
    # Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    # print()
    # print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    # print('-' * 30 + 'End: test' + '-' * 30)
    # plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    # plt.show()


def manipulate_latent(model, test_directory, image_size, args):
    print('-' * 30 + 'Begin: manipulate' + '-' * 30)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_directory,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=512,
        image_size=image_size,
        shuffle=True,
        seed=1337,
        validation_split=0.2,
        subset="validation",
    )
    for (x_test, y_test) in test_ds.take(1):
        print(args.digit)
        print("y_test", np.argmax(y_test))
        index = np.argmax(y_test, 0) == args.digit
        number = np.random.randint(low=0, high=sum(index) - 1)
        x, y = x_test[index][number], y_test[index][number]
        x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:, :, dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=14, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")

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

    args.epochs = 2
    args.n_class = 10
    args.glyph_size = (32, 32, 3)
    args.testing = False
    args.batch_size = 10
    args.test_address = "data/test/digit/"
    args.train_address = "data/neater/digit/"
    args.valid_address = "data/neater/digit/"
    args.save_dir = "log/"
    # args.w = "/home/ruhiii/Documents/AUT/Shardari/CapsNet/3/weights.h5"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=args.glyph_size,
                                                  n_class=args.n_class,
                                                  routings=args.routings,
                                                  batch_size=args.batch_size)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model,
              train_directory=args.train_address,
              valid_directory=args.valid_address,
              image_size=args.glyph_size[:-1],
              args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        # manipulate_latent(manipulate_model, args.test_address, image_size=args.glyph_size[:-1], args=args)
        test(model=eval_model, test_directory=args.test_address, image_size=args.glyph_size[:-1], args=args)
