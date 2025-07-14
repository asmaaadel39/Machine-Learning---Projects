# train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

from pathlib import Path

filepath = Path("d:/Dataset/framingham.csv")
data = pd.read_csv(filepath)
print(data.head())

#Handle Missing Values

print(data.isnull().sum())

edu_median = data['education'].mean()
data['education'].fillna(edu_median, inplace=True)

cigs_mean = data['cigsPerDay'].mean()
data['cigsPerDay'].fillna(cigs_mean, inplace=True)

BPmed_mean = data['BPMeds'].mean()
data['BPMeds'].fillna(BPmed_mean, inplace=True)

totchol_mean = data['totChol'].mean()
data['totChol'].fillna(totchol_mean, inplace=True)

BMI_mean = data['BMI'].mean()
data['BMI'].fillna(BMI_mean, inplace=True)

heart_rate_mean = data['heartRate'].mean()
data['heartRate'].fillna(heart_rate_mean, inplace=True)


glicose_mean = data['glucose'].mean()
data['glucose'].fillna(glicose_mean, inplace=True)

#Feature Selection

X = data[['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 
        'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
y = data['TenYearCHD']

#Training and Spiliting data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=10)

logistic = LogisticRegression()
logistic.fit(X_train, y_train)

y_pred = logistic.predict(X_test)

# Measuring score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
score = accuracy_score(y_test, y_pred)
print(score)

cm = confusion_matrix(y_test, y_pred)
print(cm)

cr = classification_report(y_test, y_pred)
print(cr)

joblib.dump(logistic, 'model.pkl')

