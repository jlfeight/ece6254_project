from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import manifold, decomposition
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib import offsetbox
import numpy as np
import pandas as pd
import random



year_range = range(2010,2017)

data = list()

#load all year ranges selected
for i, year in enumerate(year_range):
    url_string = 'https://sports-statistics.com/database/soccer-data/england-premier-league-%.2i-to-%.2i.csv' %(year,year+1)
    data.append(pd.read_csv(url_string))

#select which features we want and aggreagate data and create np array from df
feature_list = ['FTR','HomeTeam','AwayTeam','HTHG','HTAG','HS']

data[0] = data[0].dropna()
encoder = OneHotEncoder(sparse_output=False)
data[0] = data[0].filter(items=feature_list)
categorical_columns = data[0].select_dtypes(include=['object']).columns.tolist()
one_hot_encoded = encoder.fit_transform(data[0][categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
data_encoded = pd.concat([data[0], one_hot_df], axis=1)
data_encoded = data_encoded.drop(categorical_columns, axis=1)
data_encoded = data_encoded.dropna()
clean_data = data_encoded.to_numpy()



for nyear in range(1,len(data)):
    data[nyear] = data[nyear].dropna()
    data[nyear] = data[nyear].filter(items=feature_list)
    #categorical_columns = data[nyear].select_dtypes(include=['object']).columns.tolist()
    one_hot_encoded = encoder.fit_transform(data[nyear][categorical_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    data_encoded = pd.concat([data[nyear], one_hot_df], axis=1)
    data_encoded = data_encoded.drop(categorical_columns, axis=1)
    data_encoded = data_encoded.dropna()
    clean_data = np.vstack((clean_data,data_encoded.to_numpy()))


    
X = clean_data[:,1:]
y = clean_data[:,0]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)



n_reducedfeatures = 2

pca = decomposition.PCA(n_components=n_reducedfeatures)
pca.fit(X)
x_PCA = pca.transform(X)
