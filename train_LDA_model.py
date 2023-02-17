import pickle
import numpy as np
import pandas as pd
import gensim
import warnings

warnings.filterwarnings('ignore')
from gensim import corpora, models

dictionary = gensim.corpora.Dictionary.load('./models/dictionary.gensim')
corpus = pickle.load(open('./models/corpus.pkl', 'rb'))
df = pd.read_csv(r'./data/data_cleaned.csv')
# test
temp_ = df.shape[0]
print(temp_)
# start training
ldamodel_3 = gensim.models.ldamodel.LdaModel(corpus, num_topics=4, id2word=dictionary, passes=20)
ldamodel_3.save('./models/model.gensim')

doc_topic_max = []
for doc in ldamodel_3.get_document_topics(corpus):
    doc_topic = []
    for topic in doc:
        doc_topic.append(topic[1])  # probability
    doc_topic_max.append(doc_topic.index(max(doc_topic)))  # get the index that has most frequency
print(len(doc_topic_max))
df['topic'] = doc_topic_max[:temp_]
df.to_csv('./result/result_all.csv')

df_res = pd.DataFrame()
# according to the perplexity, we decide to use 4 topics
for el in ldamodel_3.print_topics(num_topics=4, num_words=500):
    list_value = []

    list_word = []
    temp = str(el).split('\'')[1].split('+')
    print(el, '\n')  # 0-3
    for i in temp:
        list_word.append(str(i.split('*')[1]))  # terms
        list_value.append(i.split('*')[0])  # probability
    if df_res.empty:
        df_res = pd.DataFrame(np.array([list_word, list_value])).T
    else:
        df_res = pd.concat([df_res, pd.DataFrame(np.array([list_word, list_value])).T], axis=1)
df_res.to_csv('./result/result_of_topics.csv')
