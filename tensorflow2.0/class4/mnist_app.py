from PIL import Image
import numpy as np
import tensorflow as tf

model_save_path = './checkpoint/mnist.ckpt'

# 复现前向传播
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])

# 加载模型
model.load_weights(model_save_path)


preNum = int(input("执行几次识别任务:"))

for i in range(preNum):
    image_path = input("图片文件名:")
    img = Image.open(image_path)
    img = img.resize((28, 28), Image.ANTIALIAS)  # 转化为28*28
    img_arr = np.array(img.convert('L'))  # 转化为灰度图片

    # 白底黑字，转换成黑底白字，并降噪
    # img_arr = 255 - img_arr
    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 200:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0

    img_arr = img_arr / 255.0  # 归一化

    # 神经网络训练时，都是按照batch送入网络，所以给img_arr前面添加一个维度
    # img_arr:(28, 28)
    # x_predict:(1, 28, 28)
    x_predict = img_arr[tf.newaxis, ...]

    result = model.predict(x_predict)   # 前向传播
    pred = tf.argmax(result, axis=1)    # 输出最大概率值的索引
    print('\n')
    tf.print("预测数字：", pred, "\n")

