from wordcloud import WordCloud
import matplotlib.pyplot as plt


text_cn = open('res/xyj.txt', encoding='utf8').read()

# 设置各种参数生成词云，透明背景
wc_cn = WordCloud(font_path='res/Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(text_cn)


plt.imshow(wc_cn, interpolation='bilinear')
plt.axis('off')
plt.show()


wc_cn.to_file('generate/wordcloud2.png')
