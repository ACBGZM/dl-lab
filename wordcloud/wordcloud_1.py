from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = open('res/constitution.txt').read()

wc = WordCloud().generate(text)

plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

wc.to_file('generate/wordcloud1.png')