from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba


text = open('res/xyj.txt', encoding='utf8').read()

# 中文分词
text = ' '.join(jieba.cut(text))
print(text[:100])


wc = WordCloud(font_path='res/Hiragino.ttf', width=800, height=600, mode='RGBA', background_color=None).generate(text)


plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


wc.to_file('generate/wordcloud3.png')


