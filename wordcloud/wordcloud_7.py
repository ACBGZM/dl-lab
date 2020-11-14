from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import jieba.analyse


text = open('res/xyj.txt', encoding='utf8').read()

# 提取关键词和权重
freq = jieba.analyse.extract_tags(text, topK=200, withWeight=True)
print(freq[:100])
freq = {i[0]: i[1] for i in freq}


mask = np.array(Image.open('res/color_mask.png'))
wc = WordCloud(mask=mask, font_path='res/Hiragino.ttf', mode='RGBA', background_color=None).generate_from_frequencies(freq)


plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


wc.to_file('generate/wordcloud7.png')