import tensorflow as tf
import numpy as np
from data_loader import *
from yolo_v3_model import *

# anchors_13 = np.array([[116, 90], [156, 198], [373, 326]], dtype=np.float32) / 416
#
#
# def decode_predictions(y_pred, anchors, grid_size, num_classes=3):
#     """
#     Chuyển đổi tensor đầu ra thô của model thành tọa độ bounding box thực tế.
#
#     Args:
#         y_pred (tf.Tensor): Tensor đầu ra từ một head của model.
#                              Shape: (batch_size, grid_size, grid_size, num_anchors * (5 + num_classes))
#         anchors (np.ndarray): Mảng chứa các anchor box cho head này.
#                               Shape: (num_anchors, 2)
#         grid_size (int): Kích thước của grid (ví dụ: 13, 26, hoặc 52).
#         num_classes (int): Số lượng lớp đối tượng.
#
#     Returns:
#         tf.Tensor: Tensor chứa các thông tin đã giải mã.
#                    Mỗi hàng có dạng [x_center, y_center, width, height, confidence, class_prob_1, class_prob_2, ...].
#                    Shape: (batch_size, grid_size * grid_size * num_anchors, 5 + num_classes)
#     """
#     # Lấy các thông số từ shape của tensor
#     batch_size = tf.shape(y_pred)[0]
#     num_anchors = len(anchors)
#
#     # Reshape đầu vào để dễ xử lý hơn
#     # Từ (batch, grid, grid, 24) -> (batch, grid, grid, 3, 8)
#     y_pred = tf.reshape(y_pred, (batch_size, grid_size, grid_size, num_anchors, 5 + num_classes))
#
#     # Tách các thành phần từ tensor dự đoán
#     # Các giá trị này đều là logits (đầu ra thô)
#     tx_ty = y_pred[..., 0:2]  # tx, ty
#     tw_th = y_pred[..., 2:4]  # tw, th
#     confidence_logit = y_pred[..., 4:5]
#     class_probs_logits = y_pred[..., 5:]
#
#     # --- Bước 1: Tạo Grid Cell Offsets ---
#     # Tạo một grid để biết vị trí của mỗi cell
#     grid_y = tf.tile(tf.reshape(tf.range(grid_size, dtype=tf.float32), [-1, 1, 1, 1]), [1, grid_size, 1, 1])
#     grid_x = tf.tile(tf.reshape(tf.range(grid_size, dtype=tf.float32), [1, -1, 1, 1]), [grid_size, 1, 1, 1])
#     grid = tf.concat([grid_x, grid_y], axis=-1)  # Shape: (grid_size, grid_size, 1, 2)
#
#     # --- Bước 2: Giải mã tọa độ và kích thước ---
#     # Áp dụng sigmoid cho tx, ty và cộng với offset của grid cell
#     # Sau đó chia cho grid_size để chuẩn hóa về khoảng [0, 1]
#     box_xy = (tf.sigmoid(tx_ty) + grid) / tf.cast(grid_size, tf.float32)
#
#     # Áp dụng hàm mũ cho tw, th và nhân với kích thước anchor
#     # anchors có shape (3, 2) cần reshape để nhân với tw_th có shape (batch, grid, grid, 3, 2)
#     box_wh = tf.exp(tw_th) * anchors.reshape(1, 1, 1, num_anchors, 2)
#
#     # --- Bước 3: Giải mã điểm tin cậy và xác suất lớp ---
#     confidence = tf.sigmoid(confidence_logit)
#     class_probs = tf.sigmoid(class_probs_logits)
#
#     # --- Bước 4: Ghép nối và reshape kết quả cuối cùng ---
#     decoded_preds = tf.concat([box_xy, box_wh, confidence, class_probs], axis=-1)
#
#     # Reshape thành một danh sách các dự đoán để dễ dàng xử lý sau này
#     # Shape: (batch_size, total_boxes, 5 + num_classes)
#     decoded_preds = tf.reshape(decoded_preds, (batch_size, -1, 5 + num_classes))
#
#     return decoded_preds
#
#
# images,head13,head26,head52 = datagenerator_cache()
#
# x = decode_predictions(np.expand_dims(head13[0],axis=0), anchors_13, grid_size=13, num_classes=3)
# a = 0




inference("test/images/anhthanhhoai02.jpg",3)
