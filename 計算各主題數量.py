import pandas as pd
data = pd.read_csv('dominant topics1.csv')
# tweets count
print(data.groupby('Dominant_Topic')['Text'].count())
# company count