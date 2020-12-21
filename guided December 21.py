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

text = pd.read_csv('Raw tweets_December_8:37AM_updated.csv')

# text-cleaning
b = []
for i,u in text.iterrows():
    a = []
    word =''
    for words in str(u['tweets']).split():
        if '@' not in words:
            #if u['Company'] != 'Ethicameat':
                words = words.replace('#','')
                if 'http' not in words:
                    if'&amp' not in words:
                    # words = re.sub(r'[^a-zA-Z]|(\w+:\/\/\S+)', ' ',words) #remove non-alphabets characters
                        #if u['Company'] != 'High Steaks':
                            words = re.sub(r'[^a-zA-Z]', '', words) #replace non-alphabets characters with space. From "can't" to "can t"
                            #words = words.replace(' ', '')  # remove space between contraction: From "can t" to "cant"
                            if len(words)>2:
                                if words.lower() not in stop_words:
                                    words = porter.stem(words)
                                    if len(words)>3:
                                        words = words.lower()
                                        #if 'justegg' not in words:
                                        word += (words+' ')
    b.append(word)
text['processed']=[i for i in b]
#text.to_csv('kk1.csv')


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

model = guidedlda.GuidedLDA(n_topics=6,n_iter=200,random_state=12,refresh=10,alpha=0.1,eta=0.01)
vectorizer = CountVectorizer(min_df=10)
X = vectorizer.fit_transform(text['processed'])


seed_topic_list = [['without','harm','anim','meat'],
                   ['secur','meat','consumpt','protein','altern','climat','challeng','chang','grow'],
                   ['look','hire','team','join','scientist'],
                   ['pleas','check','latest','thank','announc','stay','excit'],
                   ['breakfast','plant'],
                   ['seafood','fish','shrimp']]

vocab = vectorizer.get_feature_names()
print(vocab)
word2id = dict((v,idx) for idx,v in enumerate(vocab))
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

model.fit(X.toarray(),seed_topics=seed_topics,seed_confidence=0.35)
topic_word = model.topic_word_
n_top_words = 20
vocab = tuple(vocab)

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))




