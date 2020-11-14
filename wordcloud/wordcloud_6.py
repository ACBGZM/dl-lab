from wordcloud import WordCloud
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import jieba


text = open('res/xyj.txt', encoding='utf8').read()


text = ' '.join(jieba.cut(text))
print(text[:100])

# 颜色函数。hsl0是红色，饱和度、亮度随机
def random_color(word, font_size, position, orientation, font_path, random_state):
    s = 'hsl(0, %d%%, %d%%)' % (random.randint(60, 80), random.randint(60, 80))
    print(s)
    return s


mask = np.array(Image.open('res/color_mask.png'))
wc = WordCloud(color_func=random_color, mask=mask, font_path='res/Hiragino.ttf', mode='RGBA', background_color=None).generate(text)


plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


wc.to_file('generate/wordcloud6.png')