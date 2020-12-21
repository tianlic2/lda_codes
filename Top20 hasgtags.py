import pandas as pd
import numpy as np
import collections
data= pd.read_csv('Raw tweets_December_8:37AM.csv')
hashtag = []
date = []
for index,i in data.iterrows():
    for word in i['tweets'].split():
        if word.startswith('#'):
            date.append(i['date'])
            hashtag.append(word)
hashtag_trend = pd.DataFrame(data=hashtag,columns=['#'])
hashtag_trend['date'] = np.array(date)
data.drop_duplicates()
counts = collections.Counter(hashtag)
print(counts.most_common(20))
