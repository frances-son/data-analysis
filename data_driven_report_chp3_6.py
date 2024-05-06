## purpose ##
# colab ipython 

### clone ###
# git clone https://github.com/sangsucki/DataDrivenReport.git

### install ###
# pip install fitz
# pip install PyMuPDF
# pip install konlpy
# pip install gensim
# pip install pyLDAvis==3.4.1
# pip install pandas==1.5.1 # pandas 다운그레이드 -> LDA 시각화 위해서. (2.1.1 당시 시각화 호환이 불안정하였음)
# 에러남. pip install scipy==1.10.1로 해결 
# 한 번 실행 후 주석처리 하였음

### import ###
import gensim
import fitz
from gensim import corpora, models
import matplotlib.pyplot as plt
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models


doc = fitz.open("DataDrivenReport/문재인 대통령 연설문 선집.pdf")

# 전체 페이지 수 알기
page = doc.page_count
print(page)

texts = []
for i in range(page):
  load = doc.load_page(i)
  pagetext = load.get_text("text")
  texts.append(pagetext)

import pandas as pd
df = pd.DataFrame(texts)
from konlpy.tag import Kkma
kkma=Kkma()
print(kkma.nouns(texts[5]))

import re
dataset = []
for i in range(len(texts)) :
  dataset.append(kkma.nouns(re.sub('^가-힣a-zA-z0-9', '', texts[i])))

print(dataset)

stopwords = ['재인', '선집', '대통령', '연설문', '문재인', '여러분', '모두', '연설', '우리', '100년', '100', '기조연설', '대한민국', '대한', '민국', '기념사', '모두발언', '청와대', '신년기자회견']
data2 = [[word for word in sublist if len(word) > 1 and word not in stopwords] for sublist in dataset]

dictionary2 = corpora.Dictionary(data2)
corpus2 = [dictionary2.doc2bow(text) for text in data2]
print(corpus2)
# (0,1)의 의미 : 0번째 단어가 1번 나왔다.

random_num = 2023

coherence_values2 = []

for i in range(2,10) :
 ldamodel = gensim.models.ldamodel.LdaModel(corpus2, num_topics=i, id2word=dictionary2, random_state = random_num)
 coherence_model_lda = CoherenceModel(model=ldamodel, texts = data2, dictionary = dictionary2,topn=5)
 coherence_lda = coherence_model_lda.get_coherence()
 coherence_values2.append(coherence_lda)

coherence_values2

x = range(2,10)
plt.plot(x, coherence_values2)
plt.xlabel('number of topics')
plt.ylabel('coherence score')
# plt.show()


perplexity_values2 = []
for i in range(2,10):
  # ldamodel = gensim.models.ldamodel.LdaModel(corpus2, num_topics=i, id2word=dictionary2, random_state= random_num)
  ldamodel = gensim.models.ldamodel.LdaModel(corpus2, num_topics=i, id2word=dictionary2)
  perplexity_values2.append(ldamodel.log_perplexity(corpus2))

print(perplexity_values2)

x = range(2,10)
plt.plot(x, perplexity_values2)
plt.xlabel('number of topics')
plt.ylabel('perplexity score')
# plt.show()

ldamodel2 = gensim.models.ldamodel.LdaModel(corpus2, num_topics=6, alpha=0.1, id2word=dictionary2, random_state=random_num)

print(pd.__version__)
print(pyLDAvis.__version__)


pyLDAvis.enable_notebook()
vis2 = pyLDAvis.gensim_models.prepare(ldamodel2, corpus2, dictionary2)
# 아래 코드는 시각화가 안되어서 주석처리함. ipython 환경에서만 되는듯 함. 
# vis2

# html로 저저장하여 해결 
pyLDAvis.save_html(vis2, 'LDA_Visualization.html')

