# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:00:08 2020

@author: Venkatesh Joshi
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

df = pd.read_csv('movie_dataset.csv')

df = df.iloc[:,0:24]
print(df.columns)

#Selecting features from dataFrame
features = ['keywords','cast','genres','director']

for feature in features:
    df[feature] = df[feature].fillna(' ')

def combine_features(row):
    return row['keywords'] +" " + row['cast'] + " " + row['genres'] +" " + row['director']

df["combine_features"] = df.apply(combine_features, axis=1)

print(df.combine_features.head())

cv = CountVectorizer()

count_matrix = cv.fit_transform(df.combine_features)

cosine_sim = cosine_similarity(count_matrix)

def get_title_from_index(index):
    return df[df.index==index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title==title]["index"].values[0]

movie_user_like = input("Enter movie name")

#Get index of movie
movie_index = get_index_from_title(movie_user_like)

similiar_movies = list(enumerate(cosine_sim[int(movie_index)]))

sorted_similar_movies = sorted(similiar_movies, key=lambda x:x[1], reverse=True)

i=0
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    i = i+1
    if i>5:
        break







