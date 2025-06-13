from yolo_v3_model import create_yolo_v3
from data_loader import datagenerator, datagenerator_val
import tensorflow as tf
from losses import getloss
import numpy as np
import os

xml_list = os.listdir("data/annotations")  # lay danh sach cac file xml
xml_list_val = os.listdir("val/annotations")
batch_size = 10
epochs = 100
anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
                   dtype=np.float32) / 416

yolo_model = create_yolo_v3()
optimizer = tf.keras.optimizers.Adam(0.001)
yolo_model.compile(optimizer=optimizer, loss=[getloss(3), getloss(3), getloss(3)], run_eagerly=False)

dataset_train = tf.data.Dataset.from_generator(
    lambda: datagenerator(),
    output_signature=(
        tf.TensorSpec(shape=(416, 416, 3), dtype=tf.float32),  # Đầu vào X_batch image
        (
            tf.TensorSpec(shape=(13, 13, 3, 8), dtype=tf.float32),  # head 1
            tf.TensorSpec(shape=(26, 26, 3, 8), dtype=tf.float32),  # head 2
            tf.TensorSpec(shape=(52, 52, 3, 8), dtype=tf.float32)  # head 3
        )))
dataset_train = dataset_train.repeat()
dataset_train = dataset_train.shuffle(batch_size * 4)
dataset_train = dataset_train.batch(batch_size)
dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)


dataset_val = tf.data.Dataset.from_generator(
    lambda: datagenerator_val(),
    output_signature=(
        tf.TensorSpec(shape=(416, 416, 3), dtype=tf.float32),  # Đầu vào X_batch image
        (
            tf.TensorSpec(shape=(13, 13, 3, 8), dtype=tf.float32),  # head 1
            tf.TensorSpec(shape=(26, 26, 3, 8), dtype=tf.float32),  # head 2
            tf.TensorSpec(shape=(52, 52, 3, 8), dtype=tf.float32)  # head 3
        )))
dataset_val = dataset_val.batch(batch_size)
dataset_val = dataset_val.prefetch(tf.data.experimental.AUTOTUNE)

yolo_model.fit(dataset_train, epochs=epochs, steps_per_epoch=(len(xml_list) // batch_size), validation_data=dataset_val, validation_steps=len(xml_list_val)//batch_size)
yolo_model.save("model.h5")