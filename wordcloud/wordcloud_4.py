from wordcloud import WordCloud
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import jieba


text = open('res/xyj.txt', encoding='utf8').read()


text = ' '.join(jieba.cut(text))
print(text[:100])

# 蒙版
mask = np.array(Image.open('res/black_mask.png'))
wc = WordCloud(mask=mask, font_path='res/Hiragino.ttf', mode='RGBA', background_color=None).generate(text)


plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


wc.to_file('generate/wordcloud4.png')


