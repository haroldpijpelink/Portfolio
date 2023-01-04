#let's code a bunch of Supervised learning models and see which one works best
import pandas as pd
import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.datasets import load_wine

import numpy as np

import streamlit as st

st.title('Random Sklearn wine classifier')
#:load data


df = load_wine(as_frame=True)["frame"]
df = df.sample(frac=1)
print(df.shape)
#train test split
rng = np.random.default_rng(12345)
test_indices = rng.choice(df.index.to_numpy(), round(df.shape[0]/1.2))
test = df.iloc[test_indices]
train = df.drop(test_indices, axis =0)

train_y = train['target']
test_y = test['target']

train_X = train.drop('target', axis = 1)
test_X = test.drop('target', axis = 1)

test_scores = []

#TODO  Set up pipeline:
#model_dict = {"Gradient Boost": xgb.XGBClassifier()}
model_dict = {"Linear regression": LinearRegression(), "SVM": SVC()}
for model in model_dict.values():
    pipeline = Pipeline([
        ('standard_scaler', StandardScaler()), 
        ('pca', PCA()), 
        ('model', model)
    ])

#       Train Linear Regression model 
        #Ridge
        #Lasso
        #GLS
        #Decision Tree
        #Gradient boosted Tree
        #

#Train a model and print test set accuracy

    model = pipeline.fit(train_X, train_y)


    test_scores.append(round(pipeline.score(test_X, test_y), 2))
params = pipeline.get_params()

print(list(zip(model_dict.keys(), test_scores)))

st.subheader('Score for ' + list(model_dict.keys())[np.array(test_scores).argmax()])
st.metric("Accuracy", np.array(test_scores).max())

st.subheader('Model Parameters')
st.write(params) 