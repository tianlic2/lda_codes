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
# bigram-LDA
bigram_only = []

for index, i in text.iterrows():  # LDA
    bigram_tweets = ''
    if len(i['processed'].split())>2:
        for k in range(0, len(i['processed'].split()) - 1):
            bigram_tweets += (i['processed'].split()[k] + '_' + i['processed'].split()[k + 1]+' ')
        bigram_only.append(bigram_tweets)
import guidedlda
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

model = guidedlda.GuidedLDA(n_topics=8,n_iter=200,random_state=None,refresh=20)
vectorizer = CountVectorizer(min_df=2,max_df=0.7)
X = vectorizer.fit_transform(bigram_only)
#print(X.toarray())
#model.fit(X)

'''seed_topic_list = [['climat','environmentfriendli','environ','impact','chang','environmentfriendli','environmentallyfriendli'],
                   ['cell','cellular','grown','cultur','cellbas','labgrown','cellbas','cellbasedcultur','cellcultiv','cellcultur'],
                   ['without','harm','anim','save','help','slaughter','animal','live'],
                   ['media','twittermediainternet','multimedia','check','video','blog','blogpost','youtub'],
                   ['seafood','meatseafood','seafoodi','fish','fishi','unshellfish','jellyfish'],
                   ['look','forward','hire','scientist','join','team'],
                   ['futur','first','revolutionari','revolution','revolut']]
word2id = dict((v,idx) for idx,v in enumerate(vocab))

print(vocab)
word2id = dict((v,idx) for idx,v in enumerate(vocab))
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id'''
seed_topic_list = [['climat_chang','environment_impact','environment_friendli','impact_environ','environment_issu'],
                   ['cellbas_meat','cellbas_protein','cultiv_meat','clean_meat','muscl_cell'],
                   ['meat_without','without_anim','without_kill','without_harm','anim_suffer','anim_live','anim_health'],
                   ['social_media','pleas_check'],
                   ['seafood_product','seafood_innov','enjoy_seafood','cultur_fish','clean_fish','clean_seafood','fish_meat','fish_cell','seafood_industri'],
                   ['look_forward','join_team','come_join','want_join'],
                    ['food_secur','meat_consumpt','human_consumpt'],
                   ['nutrit_equival','delici_nutriti','healthi_sustain','public_health']]
'''seed_topic_list = [['without_harm','without_kill'],
                   ['join_team']]'''
vocab = vectorizer.get_feature_names()
word2id = dict((v,idx) for idx,v in enumerate(vocab))
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id
#print(vocab)
#model.fit(X)
model.fit(X.toarray(),seed_topics=seed_topics,seed_confidence=0.4)
topic_word = model.topic_word_
n_top_words = 20
vocab = tuple(vocab)

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))