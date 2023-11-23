import streamlit as st
import pickle
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

st.title('Amazon Product recomendation system')
title = st.text_input('Recomendation')
movie_dict=pickle.load(open('reco_dict.pkl','rb'))
df=pd.DataFrame(movie_dict)

def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stem = [stemmer.stem(w) for w in tokens]
    return " ".join(stem)
tfidvectorizer = TfidfVectorizer(tokenizer=tokenize_stem)

def cosine_sim(txt1,txt2):
    tfid_matrix = tfidvectorizer.fit_transform([txt1,txt2])
    return cosine_similarity(tfid_matrix)[0][1]

def search_product(query):
    stemmed_query = tokenize_stem(query)
    #calcualting cosine similarity between query and stemmed tokens columns
    df['similarity'] = df['stemmed_tokens'].apply(lambda x:cosine_sim(stemmed_query,x))
    res = df.sort_values(by=['similarity'],ascending=False).head(10)[['Title','Description']]
    return res
if st.button("Recomended"):
    dff=search_product(title)
    st.dataframe(dff)


