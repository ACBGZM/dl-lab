import tensorflow as tf
import func


def vgg_layers(layer_list):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(every_layer).output for every_layer in layer_list]
    model = tf.keras.Model([vgg.input], outputs)
    return model


class StyleContentModel(tf.keras.models.Model):
    # 传两个参数：指定的 VGG19 进行内容、风格损失计算的层列表
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)    # 传递风格和内容的所有层，返回的outputs是包含所有层的输出的列表
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0  # 从归一化状态恢复

        preprocessed_inputs = tf.keras.applications.vgg19.preprocess_input(inputs)

        # 把输出经过vgg网络，得到所需各层的输出，分成风格和内容输出
        outputs = self.vgg(preprocessed_inputs)  # 前向传播，返回的outputs是包含所有层的输出的列表
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        style_outputs = [func.gram_matrix(every_style_output)
                         for every_style_output in style_outputs]  # 对风格层的输出计算 Gram 矩阵

        # 整合成字典返回
        # 键：层的名称；值：这一层的输出值，风格输出是gram矩阵
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}