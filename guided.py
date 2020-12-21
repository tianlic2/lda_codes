from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag
from nltk import SnowballStemmer
from gensim.models import CoherenceModel
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
#import pyLDAvis.gensim
import re
stemmer = SnowballStemmer('english')
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

text = pd.read_csv('Raw tweets_December_8:37AM.csv')

# text-cleaning
b = []
for i,u in text.iterrows():
    a = []
    word =''
    for words in str(u['tweets']).split():
        if '@' not in words:
            if '#' not in words:
                if 'http' not in words:
                    if'&amp' not in words:
                    # words = re.sub(r'[^a-zA-Z]|(\w+:\/\/\S+)', ' ',words) #remove non-alphabets characters
                        words = re.sub(r'[^a-zA-Z]', '', words) #replace non-alphabets characters with space. From "can't" to "can t"
                        #words = words.replace(' ', '')  # remove space between contraction: From "can t" to "cant"
                        if len(words)>2:
                            if words.lower() not in stop_words:
                                    words = porter.stem(words)
                                    if len(words)>3:
                                        words = words.lower()
                                        word += (words+' ')
    b.append(word)
text['processed']=[i for i in b]


#unigram
unigram = []
unigram_list = []
for index, i in text.iterrows():
    unigram=''
    for word in i['processed'].split():
        unigram+= word+' '
    unigram_list.append(unigram)


import guidedlda
import numpy as np
import gensim


from sklearn.feature_extraction.text import CountVectorizer
data = pd.read_csv('LDA_U_Processed.csv')

model = guidedlda.GuidedLDA(n_topics=7,n_iter=50,random_state=None,refresh=20)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['processed'])


seed_topic_list = [['climat','environmentfriendli','environ','impact','chang','environmentfriendli','environmentallyfriendli'],
                   ['cell','cellular','grown','cultur','cellbas','labgrown','cellbas','cellbasedcultur','cellcultiv','cellcultur'],
                   ['without','harm','anim','save','help','slaughter','animal','live'],
                   ['media','twittermediainternet','multimedia','check','video','blog','blogpost','youtub'],
                   ['seafood','meatseafood','seafoodi','fish','fishi','unshellfish','jellyfish'],
                   ['look','forward','hire','scientist','join','team'],
                   ['futur','first','revolutionari','revolution','revolut']]

vocab = vectorizer.get_feature_names()
print(vocab)
word2id = dict((v,idx) for idx,v in enumerate(vocab))
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

model.fit(X.toarray(),seed_topics=seed_topics,seed_confidence=0.15)
topic_word = model.topic_word_
n_top_words = 20
vocab = tuple(vocab)

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))




