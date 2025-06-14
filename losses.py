import tensorflow as tf
import numpy as np
from yolo_v3_model import decode_predictions

num_class = 3

ANCHORS = np.array([[[116,90], [156,198], [373,326]],
                    [[30,61], [62,45], [59,119]],
                    [[10,13], [16,30], [33,23]]],
                   dtype=np.float32) / 416

bce_loss_calculator = tf.keras.losses.BinaryCrossentropy(from_logits=True,
                                                         reduction=tf.keras.losses.Reduction.NONE)


def compute_iou_for_yolo(boxes1, boxes2):
    x1_1 = boxes1[..., 0] - boxes1[..., 2] / 2
    y1_1 = boxes1[..., 1] - boxes1[..., 3] / 2  # y1 = y - h/2
    x2_1 = boxes1[..., 0] + boxes1[..., 2] / 2  # x2 = x + w/2
    y2_1 = boxes1[..., 1] + boxes1[..., 3] / 2  # y2 = y + h/2

    x1_2 = boxes2[..., 0] - boxes2[..., 2] / 2
    y1_2 = boxes2[..., 1] - boxes2[..., 3] / 2
    x2_2 = boxes2[..., 0] + boxes2[..., 2] / 2
    y2_2 = boxes2[..., 1] + boxes2[..., 3] / 2

    x1_inter = tf.maximum(x1_1, x1_2)  # Góc trên trái x
    y1_inter = tf.maximum(y1_1, y1_2)  # Góc trên trái y
    x2_inter = tf.minimum(x2_1, x2_2)  # Góc dưới phải x
    y2_inter = tf.minimum(y2_1, y2_2)  # Góc dưới phải y

    inter_width = tf.maximum(x2_inter - x1_inter, 0)  # Đảm bảo không âm
    inter_height = tf.maximum(y2_inter - y1_inter, 0)
    inter_area = inter_width * inter_height

    area1 = boxes1[..., 2] * boxes1[..., 3]  # w * h của boxes1
    area2 = boxes2[..., 2] * boxes2[..., 3]  # w * h của boxes2

    union_area = area1 + area2 - inter_area

    iou = inter_area / (union_area + tf.keras.backend.epsilon())  # Thêm epsilon để tránh chia cho 0

    return iou


def getloss(num_class,anchors, weight = [5.0, 5.0, 0.1, 1.0]):
    def loss_function(y_true, y_pred):

        batch = tf.shape(y_pred)[0]
        grid = tf.shape(y_true)[1]

        y_pred = tf.reshape(y_pred, [batch, grid, grid, 3 ,5+num_class])
        xy  = tf.sigmoid(y_pred[...,0:2])
        wh = y_pred[...,2:4]
        c = y_pred[...,4:5]
        p = y_pred[...,5:]
        y_pred = tf.concat([xy, wh, c, p], axis=-1)

        xywhtrue_decode = decode_predictions(y_true,anchors, grid, num_class)
        xywhpred_decode = decode_predictions(y_pred,anchors, grid, num_class)

        xywhtrue_decode = tf.reshape(xywhtrue_decode, [batch, grid, grid, 3, 5+num_class])
        xywhpred_decode = tf.reshape(xywhpred_decode, [batch, grid, grid, 3, 5 + num_class])


        ious = compute_iou_for_yolo(xywhtrue_decode, xywhpred_decode)
        ious = tf.stop_gradient(ious)

        mask_object = tf.cast(y_true[...,4:5]==1.0, tf.float32)
        mask_no_object = tf.cast(y_true[...,4:5]==0, tf.float32)

        box = tf.expand_dims(tf.keras.losses.MSE(y_true[...,0:4], y_pred[...,0:4]),axis=-1) * mask_object
        box_loss = tf.math.reduce_sum(box)

        confident_all_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ious, logits=y_pred[...,4])
        confident_loss = tf.reduce_sum(tf.expand_dims(confident_all_loss,axis=-1) * mask_object)
        confident_no_loss = tf.reduce_sum(tf.expand_dims(confident_all_loss,axis=-1) * mask_no_object)


        loss_p = tf.nn.sigmoid_cross_entropy_with_logits(y_true[...,5:] , y_pred[...,5:])
        loss_P = tf.reduce_sum(loss_p * mask_object)


        loss = (weight[0]*box_loss + weight[1]*confident_loss + confident_no_loss*weight[2] + loss_P*weight[3]) / tf.cast(batch, tf.float32)
        return loss
    return loss_function




