#Importing the libraries
import pandas as pd
import numpy as np
import pylab as pl
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm
import scipy.optimize as opt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score


#Reading the Dataset
chd_data=pd.read_csv('framingham.csv')
chd_data.drop(['education'], inplace=True, axis=1)

#Removing NaN
chd_data.dropna(axis=0, inplace=True)
print(chd_data.head(), chd_data.shape)
print(chd_data.TenYearCHD.value_counts())

#Counting number of patients affected with CHD
plt.figure(figsize=(8,6))
sn.countplot(x="TenYearCHD",data=chd_data, palette="BuGn_r")
plt.show()


#Training and Testing sets
#-----------------------------
#Declaring the x and y variables
x = np.asarray(chd_data[['age','male','cigsPerDay','totChol','glucose']])
y = np.asarray(chd_data['TenYearCHD'])


#Normalizing the dataset
x = preprocessing.StandardScaler().fit(x).transform(x)

#Actually training and testing x and y sets
x_train, x_test, y_train, y_test = train_test_split(x ,y ,test_size=0.3, random_state=4)
print('Train set: ', x_train.shape, y_train.shape)
print('Test set: ', x_test.shape, y_test.shape)


#Modeling the dataset
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)

#Evaluation and Accuracy
print('')
print('Accuracy of the model in Jaccard score is :',jaccard_score(y_test, y_pred))


#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_setup = pd.DataFrame(data=cm, columns = ['Predicted:0', 'Predicted:1',], index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(9, 6))
sn.heatmap(cm_setup, annot=True, fmt='d', cmap="Greens")
plt.show()

print('The details for the confusion matrix is :')
print(classification_report(y_test, y_pred))