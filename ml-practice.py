# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 20:00:51 2020

@author: steve
"""
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.utils import shuffle

#read in data set and check the head (delimeter is ;)
#source: https://archive.ics.uci.edu/ml/datasets/Student+Performance#
data = pd.read_table("C:/Users/steve/OneDrive/Desktop/Python Files/Data/student-performance/student-mat.csv", sep = ";")
data.head()

#our aim is to predict G3:
predict = "G3"

#summarise data
data.shape
data.info()
data.describe() # look at numerical variables
data.describe(include = ["object"]) # look at categorical variables
data.isnull().values.any()

#plot all the numerical variables
data.hist(figsize = (16,20))

#look for possible predictors
G3_corr = data.corr()[predict].sort_values(ascending = False)
plt.barh(y = G3_corr.index, width = G3_corr)

data_numeric = data.select_dtypes(np.number)
fig, axs = plt.subplots(4, 4)
for i, col in enumerate(data_numeric):
    axs[i//4, i%4].scatter(data_numeric[col], data_numeric[predict], 
                           alpha = 0.3, s = 1)
    axs[i//4, i%4].set_title(col)
fig.tight_layout()

#select columns
col_names = ["G1", "G2", "G3", "studytime", "failures", "absences"]
data = data[col_names]
data.head()

#lets graph the data
for col in data:
    #we use pandas plot functionality for efficiency rather than plt.scatter()
    data.plot(kind = "scatter", x = col, y = predict)
    plt.show()

# or a pairs plot
sns.pairplot(data)


#and view the sample correlations    
corr = data.corr()
sns.heatmap(corr, annot = True)

# plot html output in jupyter notebook
corr.style.background_gradient(cmap='coolwarm').set_precision(2) #

#G2 and G3 look to be good predictors, though they are themselves correlated

#lets convert to np arrays and create a train/test split
X = np.array(data.drop(predict, axis = 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#fit the linear model
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

print('train score:', linear.score(x_train, y_train))
print('test score:', linear.score(x_test, y_test))
