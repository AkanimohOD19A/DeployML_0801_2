from django.shortcuts import render

# import pandas as pd
# import numpy as np
import pickle

# from sklearn import linear_model
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
# from sklearn.svm import SVC, NuSVC
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


# Create your views here.

def index(request):
    return render(request, 'SNAPredictions/index.html')


def predict_sna(request):
    return render(request, 'SNAPredictions/predict_sna.html')


def result(request):
    # calling our parameters from the PRediction page
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])

    pickle_file = "C:/Users/HP/Desktop/TEST DataAnalysis/Test0801_2/DataAnalysis/model.pk2_0801_sna"
    with open(pickle_file, 'rb') as f:
        model = pickle.load(f)

    pred = model.predict([[val1, val2, val3]])
    if pred == [1]:
        result_show = "Individual is likely to Purchase"
    else:
        result_show = "Individual is NOT likely to Purchase"

    context = {'result_show': result_show}

    return render(request, 'SNAPredictions/predict_sna.html', context)
