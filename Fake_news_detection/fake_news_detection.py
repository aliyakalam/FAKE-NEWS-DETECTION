# #Importing the libraries
# import pandas as pd
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfVectorizer
# import pickle
# import nltk
# nltk.download()

# #Importing the cleaned file containing the text and label
# news = pd.read_csv('news.csv')
# X = news['text']
# y = news['label']

# #Splitting the data into train
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# #Creating a pipeline that first creates bag of words(after applying stopwords) & then applies Multinomial Naive Bayes model
# pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
#                     ('nbmodel', MultinomialNB())])

# #Training our data
# pipeline.fit(X_train, y_train)

# #Predicting the label for the test data
# pred = pipeline.predict(X_test)

# #Checking the performance of our model
# print(classification_report(y_test, pred))
# print(confusion_matrix(y_test, pred))

# #Serialising the file
# with open('model.pickle', 'wb') as handle:
#     pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
import pickle
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
news_dataset = pd.read_csv('train.csv')
news_dataset = news_dataset.fillna('')
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']
port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content 
news_dataset['content'] = news_dataset['content'].apply(stemming)
X = news_dataset['content'].values
Y = news_dataset['label'].values
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)
X_new = X_test[3]
prediction = model.predict(X_new)
print(prediction)
print(type(X_test[3]))
if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')  
with open('model.pickle', 'wb') as handle:
     pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)








