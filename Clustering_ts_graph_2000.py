
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import random
import TimeSeries_Clustering
from tqdm import tqdm 
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import sigma_gak, cdist_gak
from sklearn.metrics.cluster import adjusted_rand_score
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
import xgboost
import gc
import Evrecsys
import sys
from itertools import combinations, groupby
from collections import Counter
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import Apriori
import lightfm_form
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm import LightFM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from statistics import mode
import networkx as nx
from community import community_louvain
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor


# In[2]:


def create_item_emdedding_distance_matrix(model,interactions):
    df_item_norm_sparse = sparse.csr_matrix(model.item_embeddings)
    similarities = cosine_similarity(df_item_norm_sparse)
    item_emdedding_distance_matrix = pd.DataFrame(similarities)
    item_emdedding_distance_matrix.columns = interactions.columns
    item_emdedding_distance_matrix.index = interactions.columns
    return item_emdedding_distance_matrix

def create_user_dict(interactions):
    '''
    Function to create a user dictionary based on their index and number in interaction dataset
    Required Input - 
        interactions - dataset create by create_interaction_matrix
    Expected Output -
        user_dict - Dictionary type output containing interaction_index as key and user_id as value
    '''
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0 
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict

def create_item_dict(df,id_col,name_col):
    '''
    Function to create an item dictionary based on their item_id and item name
    Required Input - 
        - df = Pandas dataframe with Item information
        - id_col = Column name containing unique identifier for an item
        - name_col = Column name containing name of the item
    Expected Output -
        item_dict = Dictionary type output containing item_id as key and item_name as value
    '''
    item_dict ={}
    for i in range(df.shape[0]):
        item_dict[(df.loc[i,id_col])] = df.loc[i,name_col]
    return item_dict

def runMF(interactions, n_components, loss, epoch,n_jobs, item_features):
    x = sparse.csr_matrix(interactions.values)
    user_features=sparse.csr_matrix(user_features.values)
    item_features=sparse.csr_matrix(item_features.values)
    model = LightFM(no_components= n_components, loss=loss,learning_schedule='adagrad')
    model.fit(x,epochs=epoch,num_threads = n_jobs, user_features=user_features,
          item_features=item_features)
    return model


# In[3]:


#Import Data
merchants = pd.read_csv('merchants.csv')
test = pd.read_csv('test.csv', parse_dates=["first_active_month"])
train = pd.read_csv('train.csv', parse_dates=["first_active_month"])
#historical_transactions = pd.read_csv('historical_transactions.csv', parse)
#new_merchant_transactions = pd.read_csv('new_merchant_transactions.csv')
data=pd.read_csv('data_batch/data_cards_0.csv', parse_dates=['purchase_date'])
card_list=data.card_id.unique()
test= test[test.card_id.isin(card_list)]
train= train[train.card_id.isin(card_list)]


# In[4]:


def missing_impute(df):
    for i in df.columns:
        if df[i].dtype == "object":
            df[i] = df[i].fillna("other")
        elif (df[i].dtype == "int64" or df[i].dtype == "float64"):
            df[i] = df[i].fillna(df[i].mean())
        else:
            pass
    return df


# In[5]:


# Do impute missing values for all datasets
##for df in [train, test, merchants, historical_transactions, new_merchant_transactions]:
    #missing_impute(df)
for df in [train, test, merchants]:#, data]:
    missing_impute(df)


# In[6]:


#data = historical_transactions.append(new_merchant_transactions)

le = preprocessing.LabelEncoder()
le.fit(merchants['category_1'])
merchants['category_1']=le.transform(merchants['category_1']) 

le.fit(merchants['most_recent_sales_range'])
merchants['most_recent_sales_range']=le.transform(merchants['most_recent_sales_range']) 

le.fit(merchants['most_recent_purchases_range'])
merchants['most_recent_purchases_range']=le.transform(merchants['most_recent_purchases_range']) 

le.fit(merchants['category_4'])
merchants['category_4']=le.transform(merchants['category_4']) 


# In[7]:


#number of transactions
gdf = data.groupby("card_id")
gdf = gdf["purchase_amount"].size().reset_index()
gdf.columns = ["card_id", "num_transactions"]
train = pd.merge(train, gdf, on="card_id", how="left")
test= pd.merge(test, gdf, on="card_id", how="left")


# In[8]:


#Stadistics about purchase amount in new merch
gdf = data.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_trans", "mean_trans", "std_trans", "min_trans", "max_trans"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")


# In[9]:


train["year_first"] = train["first_active_month"].dt.year
test["year_first"] = test["first_active_month"].dt.year
train["month_first"] = train["first_active_month"].dt.month
test["month_first"] = test["first_active_month"].dt.month
data["year_purch"] = data["purchase_date"].dt.year
data["month_purch"] = data["purchase_date"].dt.month
data["year_month_purch"] = data["purchase_date"].dt.strftime('%Y/%m')


# ## Clustering

# ### Time Series

# In[10]:


def make_timeseries(new_transactions):
    cross_ts = pd.crosstab(new_transactions.card_id, new_transactions.year_month_purch, values=new_transactions.purchase_amount, aggfunc='sum')
    cross_ts1 = cross_ts.fillna(0).values.tolist()
    return cross_ts1, cross_ts.index


# In[11]:


def cluster_kshape(train, test, data,return_pred, num_cluster, batch):
    ts, ts_index=make_timeseries(data)
    sum_pred_test=pd.DataFrame()
    formatted_dataset = to_time_series_dataset(ts)
    X_train, sz = TimeSeries_Clustering.normalize_data(formatted_dataset)
    ks, y_pred = TimeSeries_Clustering.k_shape(X_train, n_clusters=num_cluster)
    scores = TimeSeries_Clustering.compute_scores(ks, X_train, y_pred)
    plt.boxplot(scores)

    TimeSeries_Clustering.plot_data(ks, X_train, y_pred, sz, ks.n_clusters, centroid=True)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df['card_id'] = ts_index
    y_pred_df= y_pred_df.rename({0: 'cluster'}, axis='columns')
    y_pred_df['batch'] = batch
    y_pred_df['type_cluster']='cluster_kshape'
    return y_pred_df
   


# ### Features

# In[12]:


def cluster_features(train, test, data,return_pred, num_cluster, batch):
    random_state=2
    train_test=train.append(test)
    train_test1=train_test.drop(['card_id', 'first_active_month', 'target'], axis=1)
    y_pred_test = KMeans(n_clusters=num_cluster, random_state=random_state).fit_predict(train_test1)
    y_pred_test_df = pd.DataFrame(y_pred_test)
    y_pred_test_df['card_id'] = train_test.card_id.values
    y_pred_test_df= y_pred_test_df.rename({0: 'cluster'}, axis='columns')
    y_pred_test_df['batch'] = batch
    y_pred_test_df['type_cluster']='cluster_features'
    return y_pred_test_df


# ### Graph

# In[13]:


def cluster_graph(train, test, data,return_pred, num_cluster, batch):
    FG = nx.from_pandas_edgelist(data, source='card_id', target='merchant_id', edge_attr=True)
    parts = community_louvain.best_partition(FG)
    y_pred_df = pd.DataFrame.from_dict(parts, orient='index', columns=['cluster']).reset_index()
    y_pred_df=y_pred_df.rename({'index': 'card_id'}, axis='columns')
    y_pred_df['batch']=batch
    y_pred_df['type_cluster']='cluster_graph'
    return y_pred_df
    


# ### Graph+TS

# In[14]:


def graph_kshape(train, test, data,return_pred, num_cluster, batch):
    total_cluster=pd.read_csv('results_2000/clusters_batch2000_25.csv', index_col=[0])
    cluster_card_total=pd.pivot_table(total_cluster,index=["card_id"],values=['cluster'],
               columns=['type_cluster'],aggfunc=[np.mean]).dropna()
    cluster_card_total.columns=cluster_card_total.columns.droplevel(0).droplevel(0)
    cluster_card_total=cluster_card_total.reset_index()
    cluster_card_total['cluster_graph_kshape']=cluster_card_total['cluster_graph'].map(str)+'-'+cluster_card_total['cluster_kshape'].map(str)
    y_pred_df=cluster_card_total[['card_id','cluster_graph_kshape']]
    y_pred_df=y_pred_df.rename({'cluster_graph_kshape': 'cluster'}, axis='columns')
    y_pred_df['batch']=batch
    y_pred_df['type_cluster']='cluster_graph_ts'
    return y_pred_df


# ### Graph+features

# In[15]:


def graph_features(train, test, data,return_pred, num_cluster, batch):
    total_cluster=pd.read_csv('results_2000/clusters_batch2000_25.csv', index_col=[0])
    cluster_card_total=pd.pivot_table(total_cluster,index=["card_id"],values=['cluster'],
               columns=['type_cluster'],aggfunc=[np.mean]).dropna()
    cluster_card_total.columns=cluster_card_total.columns.droplevel(0).droplevel(0)
    cluster_card_total=cluster_card_total.reset_index()
    cluster_card_total['cluster_graph_features']=cluster_card_total['cluster_graph'].map(str)+'-'+cluster_card_total['cluster_features'].map(str)
    y_pred_df=cluster_card_total[['card_id','cluster_graph_features']]
    y_pred_df=y_pred_df.rename({'cluster_graph_features': 'cluster'}, axis='columns')
    y_pred_df['batch']=batch
    y_pred_df['type_cluster']='cluster_graph_features'
    return y_pred_df


# ## Methods

# In[16]:


cols_to_use=['feature_1', 'feature_2', 'feature_3',
        'num_transactions', 'sum_trans', 'mean_trans',
       'std_trans', 'min_trans', 'max_trans', 'year_first', 'month_first']
target_col=['target']


# In[17]:


##XGBOOST

def do_xgboost(train, test, data,return_pred, num_cluster):
    param = {'max_depth': 10,
                 'eta': 0.02,
                 'colsample_bytree': 0.4,
                 'subsample': 0.75,
                 'silent': 1,
                 'nthread': 27,
                 'eval_metric': 'rmse',
                 'objective': 'reg:linear',
                 'tree_method': 'hist'
                 }
    
    X_train=train[cols_to_use]
    X_test=test[cols_to_use]
    y_train=train[target_col].values
    y_test=test[target_col].values
    
    dtrain = xgboost.DMatrix(X_train, label=y_train)
    dtest = xgboost.DMatrix(X_test)
    model = xgboost.train(param, dtrain)
    predict_test = pd.DataFrame({"card_id":test["card_id"].values})
    predict_test["target"] = pd.DataFrame(model.predict(dtest))
    if return_pred==0:
        return np.sqrt(mean_squared_error(predict_test.target.values, y_test)), predict_test
    else:
        return np.sqrt(mean_squared_error(predict_test.target.values, y_test)), predict_val, y_val, predict_test


# In[18]:


## Last_order

def avg_merchant(train, test, data,return_pred, num_cluster):
    train=train.merge(data, on='card_id', how='left')
    train_avgtarget = train.groupby(["merchant_id"])["target"].aggregate("mean").reset_index()
    train=train.merge(train_avgtarget, on='merchant_id', how='left')
    test=test.merge(data, on='card_id', how='left')
    test=test.merge(train_avgtarget, on='merchant_id', how='left').fillna(train_avgtarget.target.mean())
    pred=test.groupby(["card_id"])["target_y"].aggregate("mean").reset_index()
    pred=pred.merge(test[['card_id','target_x']], on='card_id',how='left').drop_duplicates()
    if return_pred==0:
        return np.sqrt(mean_squared_error(pred.target_y, pred.target_x)), pred
    else:
        return np.sqrt(mean_squared_error(pred.target_y, pred.target_x)), predict_val, y_val, pred


# In[19]:


## Lightfm

def do_lightfm(train, test, data,return_pred, num_cluster):
    train=train.merge(data, on='card_id', how='left')
    train_avgtarget = train.groupby(["merchant_id"])["target"].aggregate("mean").reset_index()
    train=train.merge(train_avgtarget, on='merchant_id', how='left')
    test=test.merge(data, on='card_id', how='left')
    test=test.merge(train_avgtarget, on='merchant_id', how='left')
    test_train=train.append(test)
    grouped_train_test = test_train.groupby(["merchant_id", "card_id"])["target_y"].aggregate("mean").reset_index()
    interactions = lightfm_form.create_interaction_matrix(df=grouped_train_test,user_col='merchant_id',item_col='card_id',rating_col='target_y')
    train_unique=train.drop_duplicates(subset=['card_id'])
    test_unique=test.drop_duplicates(subset=['card_id'])
    item_features= train_unique.append(test_unique)[['feature_1', 'feature_2', 'feature_3',
       'num_transactions', 'sum_trans', 'mean_trans', 'std_trans', 'min_trans',
       'max_trans', 'year_first', 'month_first']]
    mf_model = lightfm_form.runMF(interactions=interactions,
                                  n_components=30, loss='warp', epoch=40, n_jobs=4)
    # Create User Dict
    user_dict = create_user_dict(interactions=interactions)
    # Create Item dict
    products_dict = create_item_dict(df = data.reset_index(),
                               id_col = 'card_id',
                               name_col = 'card_id')
    ## Creating item-item distance matrix
    item_item_dist = create_item_emdedding_distance_matrix(model = mf_model,
                                                       interactions = interactions)
    
    scores_rmse=pd.DataFrame(columns=['card_id', 'pred'])

    for cards in test.card_id.unique():
        recommended_items = list(pd.Series(item_item_dist.loc[cards,:].                                   sort_values(ascending = False).head(10+1).                                   index[1:10+1]))
        recommended_train=list(train_unique[train_unique.card_id.isin(recommended_items)].card_id.values)
        pred=train_unique.loc[train_unique['card_id'].isin(recommended_train)]
        scores_rmse=scores_rmse.append(
            {'card_id': cards, 'pred': pred.target_x.mean()},ignore_index=True)
    scores_rmse=scores_rmse.merge(test_unique[['card_id', 'target_x']], on='card_id')
    scores_rmse=scores_rmse.fillna(scores_rmse.pred.mean())
    if return_pred==0:
        return np.sqrt(mean_squared_error(scores_rmse.pred, scores_rmse.target_x)), scores_rmse
    else:
        return np.sqrt(mean_squared_error(scores_rmse.pred, scores_rmse.target_x)), scores_rmse

    
    


# In[20]:


## Catboost
def catboost(train, test, data,return_pred, num_cluster):
    model_cat = CatBoostRegressor(iterations=500,
                             learning_rate=0.02,
                             depth=6,
                             eval_metric='RMSE',
                             bagging_temperature = 0.9,
                             od_type='Iter',
                             metric_period = 100,
                             od_wait=50)
    X_train=train[cols_to_use]
    X_test=test[cols_to_use]
    y_train=train[target_col]
    y_test=test[target_col]
    
    model_cat.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        cat_features=np.array([0,1,2]))
    
    predict_test = pd.DataFrame({"card_id":test["card_id"].values})
    predict_test["target"] = pd.DataFrame(model_cat.predict(X_test))
    if return_pred==0:
        return np.sqrt(mean_squared_error(predict_test.target.values, y_test)), predict_test
    else:
        return np.sqrt(mean_squared_error(predict_test.target.values, y_test)), predict_val, y_val, predict_test


# In[21]:


train, test = train_test_split(train, test_size=0.3)


# ### Clustering-Methods

# In[22]:


clustering_list=[graph_features, graph_kshape]
clustering_name=['graph_features', 'graph_kshape']
methods_list=[do_xgboost, do_lightfm, catboost, avg_merchant]
methods_name=['xgboost', 'do_lightfm', 'catboost', 'avg_merchant']

batch=0
return_pred=0
sum_pred_test=pd.DataFrame()
scores_cluster=pd.DataFrame(columns=['cluster_type', 'cluster_number','method', 'rmse', 'batch'])
num_cluster=25
cluster_card_total=pd.DataFrame()
i=0

for clustering in clustering_list:
    print(clustering_name[i])
    cluster_pred=clustering(train, test, data,return_pred, num_cluster, batch)
    cluster_card_total=cluster_card_total.append(cluster_pred)
    cluster_card_total.to_csv('clusters_batch2000_25_mix.csv')

    test_k = test.merge(cluster_pred, on='card_id')
    train_k = train.merge(cluster_pred, on='card_id')
    clusters = train_k.cluster.unique()
    for cluster in clusters:
        print(cluster)
        train_j = train_k[train_k['cluster'] == cluster]
        test_j = test_k[test_k['cluster'] == cluster]
        sum_rmse=0
        j=0
        for method in methods_list:
            print(methods_name[j])
            print(method)
            if len(test_j) != 0:
                try:
                    rmse, y_pred = method(train_j,test_j, data,return_pred, num_cluster)
                except:
                    pass # doing nothing on exception
            else:
                rmse=np.nan
                y_pred=pd.DataFrame.from_dict({"card_id":[0], 'target':[0]})
            scores_cluster = scores_cluster.append(
            {'cluster_type': clustering_name[i], 'cluster_number': cluster, 'method': methods_name[j], 'rmse': rmse, 'batch':batch},
            ignore_index=True)
            scores_cluster.to_csv('scores_clustering_batch2000_25cluster_mix.csv')
            sum_pred_test=sum_pred_test.append(y_pred)
            #sum_pred_test.to_csv('scores_clustering.csv')
            j = j + 1
    i=i+1
            
            
      


# In[29]:


scores_cluster=pd.read_csv('results_2000/scores_clustering_batch2000_25cluster_mix.csv').append(pd.read_csv('results_2000/scores_clustering_batch2000_25cluster.csv'))


# In[31]:


sumary=pd.pivot_table(scores_cluster,index=["cluster_type"],values=["rmse"],
               columns=["method"],aggfunc=[np.mean])


# In[32]:


print('RMSE using  25 Clusters in batch 0')
sumary


# In[52]:


cluster_card_total=pd.read_csv('results_2000/clusters_batch2000_25_mix.csv').append(pd.read_csv('results_2000/clusters_batch2000_25.csv'))


# In[55]:


cluster_card_total.head()


# In[56]:


pd.pivot_table(cluster_card_total,index=["cluster"],values=['card_id'],columns=["type_cluster"],aggfunc='count')


# In[67]:


len(cluster_card_total[cluster_card_total['type_cluster']=='cluster_graph_features'].cluster.unique())


# In[70]:


pd.crosstab(cluster_card_total.cluster, cluster_card_total.type_cluster, margins=True)[100:]


# In[58]:


pandas.crosstab(index, columns, values=None, rownames=None, colnames=None, aggfunc=None, margins=False, margins_name='All', dropna=True, normalize=False)[source]
cluster_card_total1=pd.pivot_table(cluster_card_total,index=["card_id"],values=['cluster'],
               columns=['type_cluster'])
cluster_card_total1

