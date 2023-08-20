# Natural Language Processing (NLP)

# Library Imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import string
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nltk

# In line shell that lets us install packages
# Here, we installed the 'stopwords' package
nltk.download_shell()

# Bringing in data file to examine
messages = [line.rstrip() for line in open(
    'smsspamcollection/SMSSpamCollection')]
print(len(messages))
messages[45]

# Checking our first 10 messages
for mess_num, message in enumerate(messages[:10]):
    print(mess_num, message)
    print('\n')
# Each message is clearly labeled as either ham (normal) or spam

# Setting our data into a dataframe
messages = pd.read_csv('smsspamcollection/SMSSpamCollection',
                       sep='\t', names=['label', 'message'])
messages.head()
messages.describe()

# Looking at data description, separared by label
messages.groupby('label').describe()

# Adding a length column, which is the length of the review in characters
messages['length'] = messages['message'].apply(len)

# Plotting the length of the reviews as a histogram
messages['length'].plot.hist(bins=50)

# Plotting histogram of length, split up by the 'ham' and 'spam' labels
messages.hist(column='length', by='label', bins=60, figsize=(12, 4))

# Function to process our data and remove punctuation and stop words


def text_process(message):
    """
    1. remove punctuation
    2. remove stop words
    3. return list of clean text words
    """
    no_punc = [char for char in message if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('English')]


# Using the CountVectorizer to transform our bag of words, using our function as the analyzer
bow_transformer = CountVectorizer(
    analyzer=text_process).fit(messages['message'])

print(len(bow_transformer.vocabulary_))

# Making a sparce matrix, which shows each word that appears and how many times it appears in each message
messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('# of Non-Zero occurences: ', messages_bow.nnz)

# Measuring the sparcity of our sparce matrix we just created
# Number of non-zero occurrences over the number of occurrences (basically length * width of the matrix)
sparcity = (messages_bow.nnz /
            (messages_bow.shape[0] * messages_bow.shape[1])) * 100.0
print(format(sparcity, '.5f'))

# Incorporating TF-IDF into the model
# Term Frequency-Inverse Document Frequency (TF-IDF) and weight give a value to how important a word is to a document.
# Fitting the transformer to our bag of words
tfidf_transformer = TfidfTransformer().fit(messages_bow)

# Example weight for the word 'university'
tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]

# Transform the entire bag of words into a TF-IDF corpus
messages_tfidf = tfidf_transformer.transform(messages_bow)

# Fitting this data to a Naive Bayes model
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])
# This would result it fitting this to all our training data, which we don't want
# Let's split our data up into training and testing data
msg_train, msg_test, label_train, label_test = train_test_split(
    messages['message'], messages['label'], test_size=0.3, random_state=101)

# We're going to make a pipeline that simplified this process so we don't have to do all this additional work again
pipeline = Pipeline([('bow', CountVectorizer(analyzer=text_process)),
                    ('tfidf', TfidfTransformer()), ('classifier', MultinomialNB())])
# Fitting the pipeline to our training data
pipeline.fit(msg_train, label_train)
# Predicting on our testing data
predictions = pipeline.predict(msg_test)

# Metrics to assess performance
print(classification_report(label_test, predictions))
print(confusion_matrix(label_test, predictions))

#               precision    recall  f1-score   support

#          ham       0.96      1.00      0.98      1475
#         spam       1.00      0.65      0.79       197

#     accuracy                           0.96      1672
#    macro avg       0.98      0.83      0.88      1672
# weighted avg       0.96      0.96      0.96      1672

# [[1475    0]
#  [  68  129]]

# Overall, our model based on the Naive Bayes distribution performed pretty well
# Later, we can tweak the pipeline to use a different type of classifier to see if it get different results.
