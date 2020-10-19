#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier

def find_fibra_quota(x):
    if not x.find("FIBRA")==-1:
        ix = x.find("FIBRA")+6
        iy = x[ix:].find(" ")
        iz = x[(ix+iy+1):].find(" ")
        whatwehave = x[ix: (ix+iy+iz+1)]
        if whatwehave.find("GB")!= -1:
            return float(whatwehave[:whatwehave.find(" ")])*1000
        else:
            return float(whatwehave[:whatwehave.find(" ")])
    return 100

def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,title=None,cmap=plt.cm.Blues):
    if not title:
        if normalize:
            pass
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def get_labels(array,threshold):
    lst = []
    for i in range(len(array)):
        if array[i] >= threshold:
            lst.append(1)
        else:
            lst.append(0)
    return lst

def pred_labels(array,y_true = None, set_best_threshold=True,threshold=0.1, jumps = 1000):
    if set_best_threshold:
        print("finding best threshold..")
        max_f1 = 0
        best_threshold = 0
        for i in [e/jumps for e in list(range(0,jumps))]:
            labels = get_labels(array,i)
            f1 = f1_score(y_true,labels)
            if f1>max_f1:
                max_f1 = f1
                best_threshold = i
        print("best threshold: "+str(best_threshold))
        print("f1-score at this threshold: "+str(max_f1))
        return get_labels(array,best_threshold)
    else:
        return get_labels(array,threshold)



# ### Preprocess Data

# #### > Fill NAs
def fill_nas(main, cat_features, dtypes_df):
    for col in dtypes_df.feature:
        if col in cat_features:
            main[col].fillna("Missing",inplace=True)
        else:
            main[col] = main[col].apply(lambda x: x if (x!=None) and (x!=np.nan) and (np.isnan(x)!=True) else main[col].mean())
    return main

# #### > Transform Categorical Features
def transform_cats(main, cat_features):
    for col in cat_features:
        if col.find("_transformed1")==-1 and col.find("_clustered")==-1:
            try:
                le = LabelEncoder()
                le.fit(main[col].tolist())
                main[col+"_transformed"] = le.transform(main[col].tolist())
            except:
                print("Error: "+col)
        else:
            print(col + " is already transformed.")
    return main


# ### > GradientBoostingClassifier
bst = GradientBoostingClassifier(
    loss='deviance',
    learning_rate=0.008,
    n_estimators=423,
    criterion='friedman_mse',
    min_samples_split=60,
    min_samples_leaf=20,
    max_depth=9,
    max_leaf_nodes=21,
    n_iter_no_change=20,
    tol=0.0001)