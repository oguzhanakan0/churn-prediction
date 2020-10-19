#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import plotly
import plotly.graph_objs as go
import warnings
import os
import plotly.io as pio
from sklearn.cluster import KMeans
import numpy as np

warnings.filterwarnings('ignore')


# In[2]:


def clear_file(filename='single_factor_analysis.html'):
    with open(filename, mode = "w", encoding="utf-8") as file:
        file.write("")


# In[3]:


exec(open('./single_factor_analysis/sfa_code.py').read())


# ### Read Data & Feature Dictionary

# In[4]:


def fill_nas(main, dtypes_df):
    for col in dtypes_df.feature:
        if col in cat_features:
            main[col].fillna("Missing",inplace=True)
        else:
            main[col] = main[col].apply(lambda x: x if (x!=None) and (x!=np.nan) and (np.isnan(x)!=True) else main[col].mean())
    return main


# In[ ]:


# main = pd.read_csv("./data/train.csv")
# main.reset_index(inplace=True)
# main.rename(columns={"index":"id","Churn.30":"TARGET"},inplace=True)


# In[5]:


sfa_owner_df = pd.read_csv('feature_dictionary.csv')
cont_features = list(sfa_owner_df[sfa_owner_df['dtype2']=='cont']['feature'])
cat_features = list(sfa_owner_df[sfa_owner_df['dtype2']=='cat']['feature'])
all_features = list(sfa_owner_df['feature'])


# In[ ]:


# main = fill_nas(main,sfa_owner_df)


# ### SFA - Continuous Variables

# In[6]:


main = pd.read_csv('./data/model_data_jan.csv')
# main.reset_index(inplace=True)
# main.rename(columns={"index":"id","Churn.30":"TARGET"},inplace=True)
# main = fill_nas(main,sfa_owner_df)


# In[7]:


main.head()


# In[152]:


clear_file("single_factor_analysis_jan.html")


# In[8]:


c_values = []
graph_stats_dfs_cont = []
# with open('single_factor_analysis_jan.html', mode = "a", encoding="utf-8") as file:
#     file.write("<b>Continuous Variables Analysis")
    
for feat in cont_features:
    print("="*3+" "+feat+" "+"="*3)
    graph_stats, stats_df, minimum, maximum = sfa_cont(main, feat, 'TARGET', 6,id_column="id")

    exec(open('./single_factor_analysis/draw_table.py').read())
    pio.write_image(fig0, './single_factor_analysis/images/table/'+feat+'_table.png')

    exec(open('./single_factor_analysis/draw_graph.py').read())
    pio.write_image(fig, './single_factor_analysis/images/graph/'+feat+'_graph.png')

    c_values.append([feat,stats_df.iloc[0]['org c-value']])
    graph_stats_dfs_cont.append([feat,graph_stats])

    # file.write(pio.to_html(fig, default_width = '50%', default_height = '50%'))
    # file.write(pio.to_html(fig0))
    print ('OK for '+feat)

c_values_df = pd.DataFrame(c_values, columns=['feature','c_value'])
# c_values_df.sort_values(by='c_value',ascending=False).to_excel('./cont_feature_ranking.xlsx')
c_values_df.sort_values(by='c_value',ascending=False,inplace=True)
# file.write("<b>C-Values Table</b>")
# file.write(c_values_df.to_html())
print('Finished Continuous SFA')
graph_stats_dfs_cont = pd.DataFrame(graph_stats_dfs_cont,columns=['feature','stats_df'])


# ### SFA - Categoric Variables

# In[13]:


information_values = []
graph_stats_dfs_cat=[]
# with open('zipcode_sfa.html', mode = "a", encoding="utf-8") as file:
#     file.write("<b>Continuous Variables Analysis")

for feat in cat_features:
#     if feat.find("zip_code")!=-1 or feat.find("locality")!=-1:
        # print("="*3+" "+feat+" "+"="*3)
    graph_stats, stats_df = sfa_cat(main, feat, 'TARGET',id_column="id")

    exec(open('./single_factor_analysis/draw_table_cat.py').read())
    pio.write_image(fig0, './single_factor_analysis/images/table/'+feat+'_table.png')

    exec(open('./single_factor_analysis/draw_graph_cat.py').read())
    pio.write_image(fig, './single_factor_analysis/images/graph/'+feat+'_graph.png')
    information_values.append([feat,stats_df.iloc[0]['iv']])
    graph_stats_dfs_cat.append([feat,graph_stats])

#             file.write(pio.to_html(fig, default_width = '50%', default_height = '50%'))
#             file.write(pio.to_html(fig_cat))
    print ('OK for '+feat)
        
information_values_df = pd.DataFrame(information_values, columns=['feature','iv'])
information_values_df.sort_values(by='iv',ascending=False,inplace=True)
    # information_values_df.sort_values(by='iv',ascending=False).to_excel('./cont_feature_ranking.xlsx')
#     file.write("<b>IV Table</b>")
#     file.write(information_values_df.to_html())
print('Finished Categorical SFA')
graph_stats_dfs_cat = pd.DataFrame(graph_stats_dfs_cat,columns=['feature','stats_df'])


# In[140]:


def find_familia(x):
    iy = x.find("transformed")
    ix = x.find("clustered")
    if x.find("transformed")!=-1 or x.find("clustered")!=-1:
        return x[:(max(ix,iy)-1)]
    else:
        return x


# In[141]:


information_values_df["familia"] = information_values_df.feature.apply(find_familia)


# In[143]:


information_values_df.to_excel("ivs_jan_2.xlsx")


# ### Apply Clustering (Categorical Features)

# In[169]:


def get_clustering_scheme(graph_stats_dfs_cat, steps=2, max_clusters= 10):
    clustering_scheme = {}
    for i in graph_stats_dfs_cat.index:
        tmp = graph_stats_dfs_cat.loc[i].stats_df
        n=tmp.shape[0]
        if n>=max_clusters:
            n=max_clusters
        if n>2:
            clustering_scheme[tmp.index.name] = list(range(2,n,steps))
    return clustering_scheme


# In[170]:


def apply_kmeans(graph_stats_dfs_cat, feature, n_clusters=2):
    try:
        tmp = graph_stats_dfs_cat[graph_stats_dfs_cat.feature==feature].iloc[0].stats_df
        X = np.array([[e,f] for e,f in zip([1]*tmp.shape[0],tmp.target_ratio.values)])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        tmp["category"]=kmeans.labels_
        cluster_map = {}
        for cat in tmp.category.unique():
            cluster_map[cat]=tmp[tmp['category']==cat].index.tolist()
        print(feature+" splitted to "+str(n_clusters)+" categories.")
        return cluster_map
    except:
        print(feature+" can't be splitted to "+str(n_clusters)+" categories.")


# In[171]:


def find_cluster(x,cluster_map):
    for key,value in cluster_map.items():
        if(x in value):
            return key
    print("cant find "+str(x)+" in cluster_map")
    return -1


# In[172]:


def apply_clusters_to_data(main,graph_stats_dfs_cat,feature,n_clusters=5,suffix="_clustered"):
    cluster_map = apply_kmeans(graph_stats_dfs_cat,feature, n_clusters)
    main[feature+suffix+"_"+str(len(cluster_map.keys()))] = main[feature].apply(find_cluster,args=(cluster_map,))
    return main


# In[173]:


clustering_scheme = get_clustering_scheme(graph_stats_dfs_cat,steps=3,max_clusters=12)


# In[174]:


clustering_scheme


# In[130]:


for feature in clustering_scheme.keys():
    for i in clustering_scheme[feature]:
        print(feature + " - "+ str(i))
        main = apply_clusters_to_data(main, graph_stats_dfs_cat, feature, n_clusters=i)


# ### Apply Threshold (Continuous Features)

# In[131]:


main["total_segundos_espera_transformed"] = (main["total_segundos_espera"]>25)*1
main["total_segundos_conversacion_transformed"] = (main["total_segundos_conversacion"]>75)*1
main["ranking_transformed"] = (main["ranking"]==1)*1
main["tariff_ds_fibra_quota_transformed"] = (main["tariff_ds_fibra_quota"]==300)*1


# ### Export Final Data

# In[132]:


main.to_csv('./data/model_data_jan.csv',index=False)

