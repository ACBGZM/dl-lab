import tensorflow as tf


# 读图、尺寸标准化
def load_img(img_path):
    img_max_size = 600   # 考虑到显存，设置图片的尺寸上限

    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)  # 解码
    img = tf.image.convert_image_dtype(img, tf.float32)  # 转换格式为float32

    img_shape_ori = tf.cast(tf.shape(img)[:-1], tf.float32)  # 获取图片的二维形状（不含通道数），也转换格式成float32
    img_big_size = max(img_shape_ori)  # 获取宽、高的最大值
    scale = img_max_size / img_big_size  # 计算缩放倍数
    img_shape_new = tf.cast(img_shape_ori * scale, tf.int32)  # 计算新尺寸

    img = tf.image.resize(img, img_shape_new)
    img = img[tf.newaxis, :]
    return img


# 'lijc, lijd->lcd'
# Gram[c, d] = sum_ij (F[layer, i, j, c] * F[layer, i, j, d]) / I)
# I * J = width * height
# 计算特征图的第 c 个特征图和第 d 个特征图的 Gram 矩阵值
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('lijc, lijd->lcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


# 图片截断
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


# 定义一个损失，对图像进行平滑模糊处理
# 对图片水平、数值方向求差值（梯度）
def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return x_var, y_var

def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)

