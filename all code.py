import pyLDAvis.gensim
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import SnowballStemmer
from gensim.models import CoherenceModel
from pprint import pprint
import gensim
import gensim.corpora as corpora
import pyLDAvis.gensim
import re
import tweepy
import pandas as pd
import numpy as np
import time

consumer_key = 'YRWB6yizS63ZJUW2CpEUqS7TN'
consumer_secret = 'ZdMYuSvrCPq2qv5yQ01wpE6FgaQJwvytuYuNnkuWdoMLpjO0sg'
access_token = '1315688543784562688-UYAqRnekf0mDKoQgsZWoKiBGqlffNw'
access_token_secret = 'wvORx0dHxp9uGXXjAjZEJ1hj8eLGcckRlOlmzvxhSqAEA'
auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)
tweets = []
company_twitter_name = ['MemphisMeats','mosa_meat','_superMeat_','ModernMeadow',
                        'ethicameat','MissionBarns','BalleticFoods','CubiqF',
                        'NewAgeMeats','FinlessFoods','wildtypefoods','itsmeatable',
                        'eatGOURMEY','eatjust','shiokmeats','makingmeat','BlueNaluInc','FutureMeat1','AvantMeats',
                        'AlephFarms','biftekco','CarnivoresCreed','vowfood','LabFarmFoods']
for company_name in company_twitter_name:
    print(company_name)
    tweets += api.user_timeline(company_name,count=200,page=1,include_rts=False,tweet_mode='extended')
    for i in range(2,20):
        tweets += api.user_timeline(company_name, count=200, page=i,include_rts=False,tweet_mode='extended')
    time.sleep(10)
dat = pd.DataFrame(data=[tweet.full_text for tweet in tweets],columns=['tweets'])
dat["date"]=np.array([tweet.created_at for tweet in tweets])
dat["Company"]=np.array([tweet.user.name for tweet in tweets])
dat.to_csv("Raw tweets_December_8:37AM.csv",mode='a',index=False)

data = pd.read_csv('Raw tweets_December_8:37AM.csv')
stemmer = SnowballStemmer('english')
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

# text-cleaning
b = []
for i,u in data.iterrows():
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
data['processed']=[i for i in b]
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence),deacc=True)) #將句子變為token，生成list
data_words = list(sent_to_words(data['processed'])) #tokenization
bigram = gensim.models.Phrases(data_words,min_count=1,threshold=1)

'''mincount：兩個單詞共同出現次數小於該值，則不會被考慮為bigram，
threshold：Phrases功能中會生成一個'phase score'，超過這個score的bigram會生成在最終結果中，總的來說，mincount越小，threshold越小，最終生成的bigram越多'''
bigram_mod = gensim.models.phrases.Phraser(bigram) #生成bigram
trigram = gensim.models.Phrases(bigram[data_words],threshold=1)
'''將已生成的bigram再進行一次分析，就會生成將三個（1bigram+single word）
或四個單詞（兩個bigram）看為一個整體'''
trigram_mod = gensim.models.phrases.Phraser(trigram)
def make_trigram(texts): #對texts進行分析，生成trigram
    return [trigram[doc] for doc in texts]
def make_bigrams(texts): #生成bigram
    return [bigram[doc] for doc in texts]
data_words_bigrams = make_bigrams(data_words)
data_words_trigrams = make_trigram(data_words)
id2word = corpora.Dictionary(data_words_bigrams)
texts = data_words_bigrams
corpus = [id2word.doc2bow(text) for text in texts]
print(id2word)
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=12,random_state=10,alpha='auto',per_word_topics=True)
pprint(lda_model.print_topics(num_words=400))

# counting coherence score
def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start,limit,step):
        model=gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())
        i =1
        print(i)


    return model_list, coherence_values
limit=21; start=11; step=1
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_words_bigrams, start=start, limit=limit, step=step)
# Show graph
import matplotlib.pyplot as plt

x = range(start, limit, step)
plt.plot(x, coherence_values,label='All tweets')
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(loc='best')
plt.xticks(range(start,limit,step))
plt.show()

#visualization
vis = pyLDAvis.gensim.prepare(lda_model,corpus,id2word)
pyLDAvis.save_html(vis,'LDA_Visualization.html')

#output LDA
def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: x[1], reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)
df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data['tweets'])

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.to_csv('dominant topics1.csv')