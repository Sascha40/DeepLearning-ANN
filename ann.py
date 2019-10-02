# Artificial Neural network

# Partie 1 : préparation des données

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Partie 2 - Construire le réseau de neurones

# Importation des modules de keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialisation

classifier = Sequential()

# Ajouter la couche d'entree et une couche de sortie
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11))

#Ajouter une deuxieme couche de sortie

classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))

#Ajouter la couche de sortie

classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))

# Compiler le réseau de neurones

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Entraîner le réseau de neurones

classifier.fit(X_train, y_train, batch_size=10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred =(y_pred > 0.5)

# Prédire une observation seule

"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
Xnew = pd.DataFrame(data={
        'CreditScore': [600], 
        'Geography': ['France'], 
        'Gender': ['Male'],
        'Age': [40],
        'Tenure': [3],
        'Balance': [60000],
        'NumOfProducts': [2],
        'HasCrCard': [1],
        'IsActiveMember': [1],
        'EstimatedSalary': [50000]})
Xnew = preprocess.transform(Xnew)
Xnew = np.delete(Xnew, [0,3], 1)
new_prediction = classifier.predict(Xnew)
new_prediction = (new_prediction > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
