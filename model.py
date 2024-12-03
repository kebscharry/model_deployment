from importlib.metadata import files

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

#load csv files()
df = pd.read_csv('Iris.csv')
print(df.head())

#select independent and dependent variables
x = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] #independent var
y = df['Species'] #dependent var

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=50)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#instanciate the model
clf = RandomForestClassifier()

#fit model
clf.fit(X_train, y_train)

#make pickle file of model
pickle.dump(clf, open('model.pkl', 'wb'))
