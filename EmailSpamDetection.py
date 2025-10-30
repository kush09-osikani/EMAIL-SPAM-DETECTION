import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer #to convert text to matrix of numerical data
import streamlit as st



#Data Preprocessing
data = pd.read_csv('C:\\Users\\ICL512\\Desktop\\VS CODE WORKSPACE\\MACHINE LEARNING\\spam.csv')
print(data.head(10))
print(data.info())
print(data.shape)
data.drop_duplicates(inplace=True) #inplace is used to make changes in the original data
print(data.shape)
print(data.isnull().sum())
print(data.head())
data['Category'] = data['Category'].replace(['spam', 'ham'], ['Not Spam', 'Spam']) #replacing spam and ham with Not Spam and Spam
print(data.head())

#Splitting the data into input and output datasets
X = data['Message']
y = data['Category']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
#Converting text to matrix of numerical data
cv = CountVectorizer(stop_words='english')   #removing the common words like is, the, a, an etc
cv.fit_transform(x_train) #this the input parameter data for the model
print(cv.get_feature_names_out())
features = cv.fit_transform(x_train) #passing the input data to features

#creating the model
from sklearn.naive_bayes import MultinomialNB #used for text classification
model = MultinomialNB()
model.fit(features, y_train)

#Evaluating the model
features_test = cv.transform(x_test) #transforming the test data
print(model.score(features_test, y_test)) #checking the accuracy of the model

#predicting the output
message = cv.transform(['Congratulations! You have won a lottery of $1000']) #transforming the input message
get_result = model.predict(message) #predicting the output
print(get_result)

#Creating a function for prediction and preparing for deployment
def predict(message):
    input_message = cv.transform([message]) #transforming the input message
    get_result = model.predict(input_message) #predicting the output
    return get_result

st.header("Email Spam Detection")

#output = predict('Congratulations! You have won a lottery of $1000')
#print(output)
input_mess = st.text_input("Enter the message: ")
if st.button("Validate"):
    output = predict(input_mess)
    color = "green" if output[0] == "Not Spam" else "red"
    st.markdown(
        f'<h1 style="font-family:Comic Sans MS; font-size:36px; color:{color};">The message is {output[0]}</h1>',
        unsafe_allow_html=True
    )



