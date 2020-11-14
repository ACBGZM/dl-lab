from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import jieba


text = open('res/xyj.txt', encoding='utf8').read()


text = ' '.join(jieba.cut(text))
print(text[:100])


mask = np.array(Image.open('res/color_mask.png'))
wc = WordCloud(mask=mask, font_path='res/Hiragino.ttf', mode='RGBA', background_color=None).generate(text)

# 从图片生成颜色
img_colors = ImageColorGenerator(mask)
wc.recolor(color_func=img_colors)


plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


wc.to_file('generate/wordcloud5.png')