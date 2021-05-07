import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange
import numpy as np
import time

import func
import model
from config import *

# 解决缓存问题
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False


# 读取图片
content_path = 'res/content_denji.jpg'
style_path = 'res/20.png'

content_image = func.load_img(content_path)
style_image = func.load_img(style_path)


# 定义内容、风格损失层
content_layers = ['block5_conv1']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1', ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


# 返回extractor : {'content': content_dict, 'style': style_dict}
extractor = model.StyleContentModel(style_layers, content_layers)

# 把内容、风格图片都投入模型中，输出取内容图片的内容、风格图片的风格，作为两个 target
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']


# 定义损失函数
def style_content_loss(outputs):
    style_outputs = outputs['style']        # style相关的网络层的输出，已经计算过Gram矩阵
    content_outputs = outputs['content']    # content相关的网络层的输出，不需计算Gram矩阵

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss /= num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss /= num_content_layers

    loss = style_weight * style_loss + content_weight * content_loss  # 加权求和
    return loss


# 定义优化器
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# 定义可训练对象：内容图片，添加随机噪声
# 为了加速运算，直接以内容图片作为迭代的起点，而不是随机噪声图片
image = tf.Variable(content_image + tf.random.truncated_normal(content_image.shape, mean=0.0, stddev=0.08))


# 优化过程
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        # loss += total_variation_weight * func.total_variation_loss(image)  # 加上进行平滑处理的损失

    grad = tape.gradient(loss, image)       # 求梯度
    opt.apply_gradients([(grad, image)])    # 梯度下降
    image.assign(func.clip_0_1(image))      # 01截断处理，优化让值超过范围


# 训练
for n in trange(epochs * steps_per_epoch):
    train_step(image)


plt.imshow(image.read_value()[0])
plt.show()

print(image.read_value()[0].shape)

Eimg = tf.image.convert_image_dtype(image.read_value()[0], tf.uint8)
Eimg = tf.image.encode_png(Eimg)
tf.io.write_file('/result/denji_20_bupinghua_highstyle.png', Eimg)





