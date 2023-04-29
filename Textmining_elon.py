# -*- coding: utf-8 -*-


#importing required librarires

# pip install textblob
# pip install wordcloud
# pip install spacy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.tokenize import TweetTokenizer
import string
import nltk
import re
from tqdm.notebook import tqdm_notebook
import en_core_web_sm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from matplotlib.pyplot import imread
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords') 

df = pd.read_csv("D:\\Assignments\\Text Mining\\Elon_musk.csv",encoding='latin-1')
df.head()

df["Text"]
stop_words=stopwords.words("english")
print(stop_words)

# Cleaning Tweets
tweet_one = df.iloc[4]["Text"]

def TweetCleaning(tweets):
    cleantweet = re.sub(r"@[a-zA-Z0-9]+"," ",tweets)
    cleantweet = re.sub(r"#[a-zA-Z0-9]+"," ",cleantweet)
    cleantweet=''.join(word for word in cleantweet.split() if word not in stop_words)
    return cleantweet

def calpolarity(tweets):
    return TextBlob(tweets).sentiment.polarity

def calSubjectivity(tweets):
    return TextBlob(tweets).sentiment.subjectivity


def segmentation(tweets):
    if tweets > 0:
        return "positive"
    if tweets== 0:
        return "neutral"
    else:
        return "negative"


df["cleanedtweets"]=df['Text'].apply(TweetCleaning)
df['polarity']=df["cleanedtweets"].apply(calpolarity)
df['subjectivity']=df["cleanedtweets"].apply(calSubjectivity)
df['segmentation']=df["polarity"].apply(segmentation)

df.head()

# Analysis and visualization
df.pivot_table(index=['segmentation'],aggfunc={"segmentation":'count'})


# Top three positive tweets
df.sort_values(by=['polarity'],ascending=False).head(3)

# Top three negative tweets
df.sort_values(by=['polarity'],ascending=True).head(3)

# Top three neutral tweets
df['polarity']==0
df[df['polarity']==0].head(3)

df["cleanedtweets"]

# Joining the list into one string/text
text = ' '.join(df["cleanedtweets"])
text

# Punctuation
no_punc_text = text.translate(str.maketrans('', '', string.punctuation)) 
no_punc_text

# Tokenization
from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(no_punc_text)
print(text_tokens[0:50])

# Removing stopwords
my_stop_words = stopwords.words('english')
no_stop_tokens = [word for word in text_tokens if not word in my_stop_words]

# Noramalize the data
lower_words = [x.lower() for x in no_stop_tokens]
print(lower_words[0:40])

# Stemming the data
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in lower_words]
print(stemmed_tokens[0:10])

#!python -m spacy download en
import spacy
nlp=spacy.load("en_core_web_sm")
# lemmas being one of them, but mostly POS, which will follow later
doc = nlp(' '.join(no_stop_tokens))
print(doc[0:40])


lemmas = [token.lemma_ for token in doc]
print(lemmas[0:25])

# Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(stemmed_tokens)
pd.DataFrame.from_records([vectorizer.vocabulary_]).T.sort_values(0,ascending=False).head(20)
print(vectorizer.vocabulary_)
print(vectorizer.get_feature_names()[50:100])
print(X.toarray()[50:100])
print(X.toarray().shape)

# Bigrams and Trigrams 
vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(1,3),max_features = 100)
bow_matrix_ngram =vectorizer_ngram_range.fit_transform(df["cleanedtweets"])
bow_matrix_ngram
print(vectorizer_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())


# TFidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_n_gram_max_features = TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(1,3), max_features = 10)
tf_idf_matrix_n_gram_max_features =vectorizer_n_gram_max_features.fit_transform(df["cleanedtweets"])
print(vectorizer_n_gram_max_features.get_feature_names())
print(tf_idf_matrix_n_gram_max_features.toarray())

# Wordcloud
import matplotlib.pyplot as plt
# %matplotlib inline
from wordcloud import WordCloud

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(15, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");

wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2').generate(text)
plot_cloud(wordcloud)
plt.show()

#==================
#For sentimental analysis

tweets1 = pd.read_csv(r"D:\Assignments\\Text Mining\\Elon_musk.csv",encoding=('latin-1'))
tweets1.drop(['Unnamed: 0'],inplace=True,axis=1)
tweets1.rename({'Text':'Tweets'},axis=1,inplace=True)
tweets1

tweets1 = [Tweets.strip() for Tweets in tweets1.Tweets] # remove both the leading and the trailing characters
tweets1 = [Tweets for Tweets in tweets1 if Tweets] # removes empty strings, because they are considered in Python as False 


import spacy
from nltk import tokenize
sentences = tokenize.sent_tokenize(" ".join(tweets1))
sentences[5:50]


text_df = pd.DataFrame(sentences, columns=['text'])
text_df


affinity_scores = text_df.set_index('text').to_dict() 
affinity_scores


#Custom function :score each word in a sentence in lemmatised form, 
#but calculate the score for the whole original sentence.
nlp = spacy.load('en_core_web_sm')
sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)
    return sent_score 

# test that it works
calculate_sentiment(text = 'tweets')

text_df['text'][:5].apply(lambda x: TextBlob(x).sentiment)


text_df['sentiment'] = text_df['text'].apply(lambda x: TextBlob(x).sentiment[0] )
text_df[['text','sentiment']]



import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(text_df['sentiment']) 


plt.figure(figsize=(15,10))
plt.xlabel('index')
plt.ylabel('sentiment')
sns.lineplot(data=text_df) 





























