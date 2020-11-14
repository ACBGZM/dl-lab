from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 英文文本
text_en = open('res/constitution.txt').read()

# 生成词云
wc_en = WordCloud().generate(text_en)


plt.imshow(wc_en, interpolation='bilinear')
plt.axis('off')
plt.show()


wc_en.to_file('generate/wordcloud2.png')