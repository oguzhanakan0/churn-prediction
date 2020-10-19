#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[30]:


def find_fibra_quota(x):
    if not x.find("FIBRA")==-1:
        ix = x.find("FIBRA")+6
        iy = x[ix:].find(" ")
        iz = x[(ix+iy+1):].find(" ")
        whatwehave = x[ix: (ix+iy+iz+1)]
        if whatwehave.find("GB")!= -1:
            return float(whatwehave[:whatwehave.find(" ")])*1000
        else:
            try:
                return float(whatwehave[:whatwehave.find(" ")])
            except ValueError as e:
                print(x,end=" :")
                print(e)
                print("tried to convert: "+whatwehave)
                print("returning this: "+str(float(whatwehave)*1000))
                return float(whatwehave)*1000
    return 100


# ### Read Data

# In[93]:


main = pd.read_csv("./data/train.csv")
main_test = pd.read_csv("./data/test.csv")
main = pd.concat([main,main_test],axis=0)


# In[136]:


main = pd.read_csv("./data/train_feb.csv")


# ### Define Columns

# In[137]:


target = "Churn.30"
# target = "TARGET"
relevant_columns = main.columns.tolist()
relevant_columns.remove(target)


# ### Exploring the columns

# In[63]:


for col in relevant_columns:
    print("="*5,end=" ")
    print(col,end=" ")
    print("="*5)
    print(main[col].value_counts(dropna=False))


# ### Correct type of the columns

# In[138]:


main["grupo_acd"] = main["grupo_acd"].astype(str)
main["locality_area"] = main["locality_area"].astype(str)
main["outcome_baja"] = main["outcome_baja"].astype(str)
main["zip_code"] = main["zip_code"].astype(str)
main["times_retention"] = main["times_retention"].astype(str)


# ### Create New Features

# In[139]:


main["client_segment"] = main.grupo_acd.apply(lambda x: x[-3:])
main["bpo"] = main.grupo_acd.apply(lambda x: x[:2])


# In[140]:


main["tariff_ds_has_fibra"] = main.tariff_ds.apply(lambda x: x.find("FIBRA")!=-1)
main["tariff_ds_fibra_quota"] = main.tariff_ds.apply(find_fibra_quota)


# ### Export Feature Dictionary & Final Data

# In[141]:


dtypes_ = dict(zip(main.dtypes.index,main.dtypes))
dtypes_.pop(target)
dtypes_.pop('id_agente')
# dtypes_.pop('id')
# dtypes_.pop('TARGET')
# dtypes_.pop('id')


# In[142]:


dtypes_df = pd.DataFrame.from_dict(dtypes_,orient="index",columns=["dtype"])
dtypes_df.reset_index(inplace=True)
dtypes_df.rename(columns={"index":"feature"},inplace=True)
dtypes_df["dtype2"] = dtypes_df.dtype.apply(lambda x: "cont" if x in [np.dtype("int64"),np.dtype("float64")] else "cat")
dtypes_df["dtype2"] = dtypes_df.apply(lambda x: "cat" if (x.feature.find("_transformed")!=-1 or x.feature.find("_clustered")!=-1) else x.dtype2, axis=1)


# In[143]:


dtypes_df["distinct_count"] = dtypes_df.feature.apply(lambda x: main[x].value_counts(dropna=False).shape[0]) 
dtypes_df["na_count"] = dtypes_df.feature.apply(lambda x: main[main[x].isna()].shape[0])
dtypes_df["na_perc"] = dtypes_df.na_count.apply(lambda x: x/main.shape[0])


# In[144]:


dtypes_df.to_csv('feature_dictionary_feb.csv')


# In[153]:


def fill_nas(main):
    for col in dtypes_df.feature:
        try:
            if col in cat_features:
                main[col].fillna("Missing",inplace=True)
            else:
                main[col] = main[col].apply(lambda x: x if (x!=None) and (x!=np.nan) and (np.isnan(x)!=True) else main[col].mean())
        except:
            print(col+" cannot be found (or something else happened.)")
    return main


# In[154]:


main = fill_nas(main)


# In[168]:


main.to_pickle('./data/model_data_feb.pkl')

