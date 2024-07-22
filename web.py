import streamlit as st
import pickle
import pandas as pd
import re

st.set_page_config(layout="wide")

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title('Fake News Detection System')

# function to process the text
def textFormatter(text):
    text = text.lower()  # convert string into lowercase
    text = re.sub(r'https?://\S+|www\.\S+','',text)   # remove urls
    text = re.sub(r'<.*?>','',text)    #remove HTML tags
    text = re.sub(r'[^\W\S]','',text)   #remove punctuation
    text = re.sub(r'\d', '',text)       #remove digits
    text = re.sub(r'\n', '',text)    #remove newline characters

    return text


def manual_tester(news):
    temp = {"text" : [news]}
    x_news = pd.DataFrame(temp)
    x_news["text"] = x_news["text"].apply(textFormatter)
    xv_news = vectorizer.transform(x_news['text'])
    prediction = model.predict(xv_news)

    return prediction[0]


news = st.text_area(label = '', placeholder= "Enter the news")

if st.button('Check'):
    if news == '':
        st.warning("Please enter the news")
    else:
        result = manual_tester(news)

        if result == 1:
            st.success("Real News")
        else:
            st.warning('Fake News')
