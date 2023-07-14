import pickle
import matplotlib.pyplot as plt
import numpy as np
import gensim
import warnings
warnings.filterwarnings('ignore')
from gensim import corpora, models

dictionary = gensim.corpora.Dictionary.load('./models/dictionary.gensim')
corpus = pickle.load(open('./models/corpus.pkl', 'rb'))

corpus = [x for x in corpus if len(x)>0]
pickle.dump(corpus, open('./models/corpus.pkl', 'wb'))

listr=[]
for i in range(2,8):
    # takes some time to train the model and get scores
    ldamodel_3 = gensim.models.ldamodel.LdaModel(corpus, num_topics=i, id2word = dictionary, passes=20)
    perplexity = ldamodel_3.log_perplexity(corpus)
    listr.append(perplexity)
    print(perplexity)


y = [np.exp2(-i) for i in listr]
x = [i for i in range(2,8)]
plt.plot(x,y)
# plt.grid()
# plt.title("lda Topic Number VS Perplexity")
plt.xlabel('Topic Number')
plt.ylabel('Perplexity')
plt.savefig('./perplexity.png', dpi = 600)
plt.show()
