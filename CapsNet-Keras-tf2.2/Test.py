import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_loader = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/ruhiii/PycharmProjects/plate_recognition/synthetic_plates/output/yolo/train/glyphs/",
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=32,
    image_size=(80, 80),
    shuffle=True,
    seed=1337,
    validation_split=0.2,
    subset="validation",
)
for (images, labels) in train_loader.take(1):
    print(images.shape)

# train_datagen = ImageDataGenerator(width_shift_range=0.1,
#                                    height_shift_range=0.1)  # shift up to 2 pixel for MNIST
# generator = train_datagen.flow_from_directory(
#     "/home/ruhiii/PycharmProjects/plate_recognition/synthetic_plates/output/yolo/train/glyphs/",
#     batch_size=32)
# X = generator.next()
# print(X.shape)
