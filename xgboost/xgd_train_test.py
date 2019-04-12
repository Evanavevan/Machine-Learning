
# coding: utf-8

# ## 建模初次训练查看结果

# # 决策树

# In[1]:


from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import pandas as pd
import numpy as np


# In[42]:


#读入数据集
df_off = pd.read_csv('train_feature.csv')
df_test = pd.read_csv('test_feature.csv')


# In[33]:


#df_test_ = pd.read_csv('ccf_offline_stage1_test_revised.csv')
#df_test_.drop_duplicates(inplace=True)


# In[43]:


df_off.drop('Date_received', axis=1, inplace=True)


# In[44]:


df_test.drop_duplicates(subset=['User_id', 'Coupon_id', 'Date_received'], inplace=True)
df_test.index = range(len(df_test))
result = df_test[['User_id', 'Coupon_id', 'Date_received']]
df_test.drop('Date_received', axis=1, inplace=True)
df_test


# In[45]:


result


# In[46]:


df_off.drop(['User_id', 'Coupon_id', 'Merchant_id'], axis=1, inplace=True)
df_test.drop(['User_id', 'Coupon_id', 'Merchant_id'], axis=1, inplace=True)


# In[47]:


df_off.drop(['user_use_same_coupon_rate', 'user_receive_same_coupon_count', 'user_use_all_coupon_rate'], axis=1, inplace=True)
df_test.drop(['user_use_same_coupon_rate', 'user_receive_same_coupon_count', 'user_use_all_coupon_rate'], axis=1, inplace=True)


# In[48]:


#df_test.Coupon_id = df_test.Coupon_id.astype(float)
df_test.user_receive_all_coupon_count = df_test.user_receive_all_coupon_count.astype(float)
#df_test.user_receive_same_coupon_count = df_test.user_receive_same_coupon_count.astype(float)
df_test.this_month_user_receive_all_coupon_count = df_test.this_month_user_receive_all_coupon_count.astype(float)
df_test.this_month_user_receive_same_coupon_count = df_test.this_month_user_receive_same_coupon_count.astype(float)
df_test.total_coupon = df_test.total_coupon.astype(float)
df_test.every_coupon_count = df_test.every_coupon_count.astype(float)


# In[49]:


df_off.info()


# In[50]:


df_test.info()


# In[51]:


#划分数据集
test = df_off.sample(frac=0.2, axis=0)
train = df_off.drop(np.array(test.index), axis=0)


# In[52]:


test_x = test.drop('label', axis=1)
test_y = test[['label']]
train_x = train.drop('label', axis=1)
train_y = train[['label']]


# In[53]:


dataset1 = xgb.DMatrix(train_x, train_y)


# In[54]:


dataset2 = xgb.DMatrix(test_x, test_y)


# In[55]:


dataset3 = xgb.DMatrix(df_test)


# In[36]:


dataset1.feature_names


# In[37]:


#df_off.isnull().sum()


# In[19]:


#df_off.label.value_counts()


# In[20]:


#df_off.drop(['User_id', 'Merchant_id', 'Coupon_id'], axis=1, inplace=True)


# In[21]:


#train_x = df_off.drop('label', axis=1)
#train_y = df_off[['label']]
#dataset1 = xgb.DMatrix(train_x, train_y)
#dataset2 = xgb.DMatrix(df_test)


# In[22]:


train_y.shape


# In[16]:


params={'booster':'gbtree',
	    'objective': 'rank:pairwise',
	    'eval_metric':'auc',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':5,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.01,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }


# In[56]:


watchlist = [(dataset1,'train'), (dataset2, 'test')]
model = xgb.train(params,dataset1,num_boost_round=600,evals=watchlist)


# In[59]:


#预测测试集概率
result['Probability'] = model.predict(dataset3)
result.label = MinMaxScaler().fit_transform(np.array(result.Probability).reshape(-1, 1))
result.sort_values(by=['Coupon_id','Probability'],inplace=True)
result.to_csv("test_predction.csv",index=None,header=None)
result.describe()


# In[57]:


#save feature score
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1}\n".format(key,value))
    
with open('xgb_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)


# In[58]:


feature_score

