# Load Libraries
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC

import warnings
warnings.filterwarnings('ignore')


# Load Dataset
sna_Df = pd.read_csv("C:/Users/HP/Desktop/TEST DataAnalysis/Test0801_2/DataAnalysis/datasets_Social_Network_Ads.csv")

sna_Df.drop(columns = 'User ID', inplace = True)

### Male == 0; Female == 1
sna_Df["Gender"] = sna_Df["Gender"].replace({'Male':0, 'Female':1})
print(sna_Df.head())

## Convert DataFrame into Array
array = sna_Df.values

X = array[:, 0:3]
y = array[:, 3]

## Test/Train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

## Trying out Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

pipe = Pipeline([('scalar', StandardScaler()),('svc', SVC())])

model = pipe.fit(X_train, y_train)

### Predictions
pred = model.predict(X_test)

val_accuracy = accuracy_score(y_test, pred)

print('Model Accuracy Score: {0:0.4f}'.format(val_accuracy))

### Testing our predictions for production
print("##############################################")
print("Testing Model Prediction")

predPro = model.predict([[0, 19, 19000]])
if predPro == [1]:
    print("Individual is likely to Purchase")
else:
    print("Individual is NOT likely to Purchase")
    
print(predPro, '\n')
print("END OF TEST")
print("##############################################")

#Check the prediction precision and accuracy
from sklearn.metrics import classification_report

print(classification_report(y_test, pred))

#Saving the model with pickle
import pickle

# save the model to disk
model_name = 'model.pk2_0801_sna'
pickle.dump(model, open(model_name, 'wb'))

print("[INFO]: Finished saving model...")

