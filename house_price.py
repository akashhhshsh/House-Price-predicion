import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('housing.csv')

#print(data)
# print(data.info())
data.dropna(inplace=True)
# print(data)
# print(data.info())

X = data.drop(['median_house_value'],axis=1)
y = data['median_house_value']

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)

train_data = X_train.join(y_train)
print(train_data)

train_data1 = train_data.drop(['ocean_proximity'],axis=1)
train_data1.hist(figsize = (15,8))
# plt.show()


#print(train_data1.corr())
plt.figure(figsize=(14,9))
sns.heatmap(train_data1.corr(),annot=True,cmap='YlGnBu')
# plt.show()

#since some features are right skewed we standardise them
train_data1['total_rooms'] = np.log(train_data1[ 'total_rooms']+1) 
train_data1['total_bedrooms'] = np.log(train_data1[ 'total_bedrooms']+1) 
train_data1['population'] = np.log(train_data1[ 'population']+1) 
train_data1['households'] = np.log(train_data1[ 'households']+1) 

train_data1.hist(figsize = (15,8))
# plt.show()

# Convert the ocean_proximity feature to a numeric format
print(train_data.ocean_proximity.value_counts())
# Convert the ocean_proximity feature to a numeric format
ocean_numeric = pd.get_dummies(train_data.ocean_proximity).astype(int)
#print(ocean_numeric)

train_data2 = train_data1.join(ocean_numeric)
print(train_data2)

#Find the correlation between all values
print(train_data2.corr())
plt.figure(figsize=(14,9))
sns.heatmap(train_data2.corr(),annot=True,cmap='YlGnBu')
plt.show()

#code completed on google colab