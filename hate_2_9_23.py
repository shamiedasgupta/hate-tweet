# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 22:06:22 2023

@author: manas
"""

# Data Analysis
import pandas as pd
import numpy as np

# Data Visualization
from matplotlib import pyplot as plt
import seaborn as sns


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
import re

# Warning
import warnings
warnings.filterwarnings('ignore')





# Loading train data
train_df = pd.read_csv(r"D:\Documents\p-a-s\train.csv\train.csv")
print(f'Train data shape: {train_df.shape}')


test_df = pd.read_csv(r"D:\Documents\p-a-s\test.csv\test.csv")
print(f'test data shape: {test_df.shape}')



train_df.duplicated().sum()
train_df.dtypes

# Missing values check
print(f'Missing values in train data:\n{train_df.isnull().sum()}')
print('-'*40)
print(f'Missing values in test data:\n{test_df.isnull().sum()}')

# Plotting wordclouds for both negative and positive tweets
stopwords = set(STOPWORDS)


stopwords.add('user')        

negative_tweets = train_df['tweet'][train_df['label']==1].to_string()
wordcloud_negative = WordCloud(width = 800, height = 800, 
                               background_color ='white', stopwords = stopwords,
                               min_font_size = 10).generate(negative_tweets)

positive_tweets = train_df['tweet'][train_df['label']==0].to_string()
wordcloud_positive = WordCloud(width = 800, height = 800, 
                               background_color ='white', stopwords = stopwords,
                               min_font_size = 10).generate(positive_tweets)
 
                     
plt.figure(figsize=(14, 6), facecolor = None)

plt.subplot(1, 2, 1)
plt.imshow(wordcloud_negative)
plt.axis("off")
plt.title('Negative Tweets', fontdict={'fontsize': 20})

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_positive)
plt.axis("off")
plt.title('Positive Tweets', fontdict={'fontsize': 20})

plt.tight_layout() 
plt.show()



# Feature Engineering
train_df_fe = train_df.copy()
train_df_fe['tweet_length'] = train_df_fe['tweet'].str.len()
train_df_fe['num_hashtags'] = train_df_fe['tweet'].str.count('#')
train_df_fe['num_exclamation_marks'] = train_df_fe['tweet'].str.count('\!')
train_df_fe['num_question_marks'] = train_df_fe['tweet'].str.count('\?')
train_df_fe['total_tags'] = train_df_fe['tweet'].str.count('@')
train_df_fe['num_punctuations'] = train_df_fe['tweet'].str.count('[.,:;]')
train_df_fe['num_question_marks'] = train_df_fe['tweet'].str.count('[*&$%]')
train_df_fe['num_words'] = train_df_fe['tweet'].apply(lambda x: len(x.split()))



# Visualizing relationship of newly created features with the tweet sentiments
plt.figure(figsize=(12, 16))
features = ['tweet_length', 'num_hashtags', 'num_exclamation_marks', 'num_question_marks', 
            'total_tags', 'num_punctuations', 'num_words']
for i in range(len(features)):
    plt.subplot(4, 2, i+1)
    sns.distplot(train_df_fe[train_df_fe.label==0][features[i]], label = 'Positive')
    sns.distplot(train_df_fe[train_df_fe.label==1][features[i]], label = 'Negative')
    plt.legend()
plt.tight_layout()
plt.show()



# Train-Test Splitting
X = train_df.drop(columns=['label'])
y = train_df['label']
test = test_df
print(X.shape, test.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Function to tokenize and clean the text
def tokenize_and_clean(text):
    # Changing case of the text to lower case
    lowered = text.lower()
    
    # Cleaning the text
    cleaned = re.sub('@user', '', lowered)
    
    # Tokenization
    tokens = word_tokenize(cleaned)
    filtered_tokens = [token for token in tokens if re.match(r'\w{1,}', token)]
    
    # Stemming
    stemmer = PorterStemmer()
    stems = [stemmer.stem(token) for token in filtered_tokens]
    return stems



# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_clean, stop_words='english')
X_train_tweets_tfidf = tfidf_vectorizer.fit_transform(X_train['tweet'])
X_test_tweets_tfidf = tfidf_vectorizer.transform(X_test['tweet'])
print(X_train_tweets_tfidf.shape, X_test_tweets_tfidf.shape)

# TF-IDF Vectorization on full training data
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_clean, stop_words='english')
X_tweets_tfidf = tfidf_vectorizer.fit_transform(X['tweet'])
test_tweets_tfidf = tfidf_vectorizer.transform(test['tweet'])
print(X_tweets_tfidf.shape, test_tweets_tfidf.shape)



plt.pie(y_train.value_counts(), 
        labels=['Label 0 (Positive Tweets)', 'Label 1 (Negative Tweets)'], 
        autopct='%0.1f%%')
plt.axis('equal')
plt.show()


# SMOTE
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train_tweets_tfidf, y_train.values)
print(X_train_smote.shape, y_train_smote.shape)

# SMOTE on full training data
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X_tweets_tfidf, y.values)
print(X_smote.shape, y_smote.shape)


plt.pie(pd.value_counts(y_train_smote), 
        labels=['Label 0 (Positive Tweets)', 'Label 1 (Negative Tweets)'], 
        autopct='%0.1f%%')
plt.axis('equal')
plt.show()

# to print scores
def training_scores(y_act, y_pred):
    acc = round(accuracy_score(y_act, y_pred), 3)
    f1 = round(f1_score(y_act, y_pred), 3)
    print(f'Training Scores: Accuracy={acc}, F1-Score={f1}')
    
def validation_scores(y_act, y_pred):
    acc = round(accuracy_score(y_act, y_pred), 3)
    f1 = round(f1_score(y_act, y_pred), 3)
    print(f'Validation Scores: Accuracy={acc}, F1-Score={f1}')



# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_smote, y_train_smote)
y_train_pred = lr.predict(X_train_smote)
y_test_pred = lr.predict(X_test_tweets_tfidf)
training_scores(y_train_smote, y_train_pred)
validation_scores(y_test, y_test_pred)






# Naive Bayes Classifier
mnb = MultinomialNB()
mnb.fit(X_train_smote, y_train_smote)
y_train_pred = mnb.predict(X_train_smote)
y_test_pred = mnb.predict(X_test_tweets_tfidf)
training_scores(y_train_smote, y_train_pred)
validation_scores(y_test, y_test_pred)

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train_smote, y_train_smote)
y_train_pred = rf.predict(X_train_smote)
y_test_pred = rf.predict(X_test_tweets_tfidf)
training_scores(y_train_smote, y_train_pred)
validation_scores(y_test, y_test_pred)


# Extreme Gradient Boosting Classifier
xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
xgb.fit(X_train_smote, y_train_smote)
y_train_pred = xgb.predict(X_train_smote)
y_test_pred = xgb.predict(X_test_tweets_tfidf)
training_scores(y_train_smote, y_train_pred)
validation_scores(y_test, y_test_pred)


#HYPERPARAMETER TUNING
# Random Forest Classifier
rf = RandomForestClassifier(criterion='entropy', max_samples=0.8, 
                            min_samples_split=10, random_state=0)
rf.fit(X_train_smote, y_train_smote)
y_train_pred = rf.predict(X_train_smote)
y_test_pred = rf.predict(X_test_tweets_tfidf)
training_scores(y_train_smote, y_train_pred)
validation_scores(y_test, y_test_pred)
# Public F1-Score = 0.727

# Extreme Gradient Boosting Classifier
xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', 
                    learning_rate=0.8, max_depth=20, gamma=0.6, 
                    reg_lambda=0.1, reg_alpha=0.1)
xgb.fit(X_train_smote, y_train_smote)
y_train_pred = xgb.predict(X_train_smote)
y_test_pred = xgb.predict(X_test_tweets_tfidf)
training_scores(y_train_smote, y_train_pred)
validation_scores(y_test, y_test_pred)
# Public F1-Score = 0.692




# Predicting test data on full training data
rf = RandomForestClassifier(criterion='entropy', max_samples=0.8, 
                            min_samples_split=10, random_state=0)
rf.fit(X_smote, y_smote)
predictions = rf.predict(test_tweets_tfidf)
submission = pd.DataFrame({'id':test_df.id,'tweet':test_df.tweet, 'label':predictions})
submission.head()



submission.to_csv(r"D:\Documents\p-a-s\test.csv\Submission.csv", index=False)
print('Submission is successful!')





