import re

import pandas as pd

import numpy as np


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv(r'data/con_devops2.csv')

is_test = False
if is_test:
    df = df[:100]

def remove_brackets(x):
    '''the function used to remove html tags
    '''
    return re.sub('<.*?>', '', str(x))

df["Body"] = df["Body"].apply(remove_brackets)
doc_set = df.loc[:,'Body'].T.tolist()[:]


import nltk
# if you are new to use stopwords in nltk, then you should download it
nltk.download("stopwords")
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
stopworddic = set(stopwords.words('english'))

listq=['br','href','b','amp','>','org','.','u','nh','bbc','viru']
tokenined_docs = []
import re

# remove characters
r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
stoplist = ['e','f','x','r','u','v','b1','j','id','pre','p','quot',
'gt','lt','xml','com','amp','image','org','http','would','i', 'me',
 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
  'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
   'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they',
    'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
     'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
      'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
       'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
        'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
         'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
          'into', 'through', 'during', 'before', 'after', 'above', 'below',
           'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there',
             'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
               'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd',
                 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn',
                  'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
                   'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
                    'weren', 'won', 'wouldn']

listr = []
list_w=[]
strq=''


count = 0
for doc in doc_set:
    count += 1
    tokenined_ = []
    temp=''
    doc = doc.lower()
    # tokenize terms and split stopwords
    list_cut = [i for i in nltk.regexp_tokenize(doc,"[\w']+") if i not in stopworddic]
    list_cut = [i for i in list_cut if i not in stoplist]
    # tense removal
    words_1=''
    for i in list_cut:
      words_1+=' '+i
    list_time = nltk.word_tokenize(text=words_1,language="english")

    # remove plural form
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    word_2 = [wordnet_lemmatizer.lemmatize('%s'%i) for i in list_time]

    # filtering
    word_ss=[]
    for sent in word_2:
      word_ss.append(nltk.pos_tag(nltk.word_tokenize(sent))[0])
    for (x,a) in word_ss:
        # obtain useful words
        if x not in stoplist and( a=='NN' or a =='VB' or a=='JJ'):
            sentence = x
            if sentence != '':
                try:
                    # become sentence
                    tokenined_.append(sentence)
                    temp+=sentence
                    strq+=' '+sentence
                except:
                    pass

    list_w.append(temp)
    tokenined_docs.append(tokenined_)

pd.DataFrame(np.array(list_w).T).to_csv('./result/data_sentence_handled.csv')
pd.DataFrame(np.array(tokenined_docs).T).to_csv('./result/data_word_handled.csv')
print(tokenined_docs[0:3])
print(len(tokenined_docs))

from gensim import corpora

# term frequency
tokens = tokenined_docs
dictionary = corpora.Dictionary(tokens)
corpus = [dictionary.doc2bow(text) for text in tokens]

#### pre-process finished

# save the terms, so we don't need to pre-process again

import pickle

pickle.dump(corpus, open('./models/corpus.pkl', 'wb'))
dictionary.save('./models/dictionary.gensim')

df.to_csv('./data/data_cleaned.csv')