
# coding: utf-8

# # XGB分类器

# In[1]:


import xgboost
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


# In[2]:


#读入数据集
df_off = pd.read_csv('train_feature.csv')
df_test = pd.read_csv('test_feature.csv')


# In[3]:


df_off.drop('Date_received', axis=1, inplace=True)
df_test.drop_duplicates(subset=['User_id', 'Coupon_id', 'Date_received'], inplace=True)
df_test.index = range(len(df_test))
result = df_test[['User_id', 'Coupon_id', 'Date_received']]
df_test.drop('Date_received', axis=1, inplace=True)


# In[4]:


df_test.drop(['User_id', 'Coupon_id', 'Merchant_id'], axis=1, inplace=True)
#df_off.drop(['user_use_same_coupon_rate', 'user_receive_same_coupon_count', 'user_use_all_coupon_rate'], axis=1, inplace=True)
#df_test.drop(['user_use_same_coupon_rate', 'user_receive_same_coupon_count', 'user_use_all_coupon_rate'], axis=1, inplace=True)


# In[5]:


#df_test.Coupon_id = df_test.Coupon_id.astype(float)
df_test.user_receive_all_coupon_count = df_test.user_receive_all_coupon_count.astype(float)
df_test.user_receive_same_coupon_count = df_test.user_receive_same_coupon_count.astype(float)
df_test.this_month_user_receive_all_coupon_count = df_test.this_month_user_receive_all_coupon_count.astype(float)
df_test.this_month_user_receive_same_coupon_count = df_test.this_month_user_receive_same_coupon_count.astype(float)
df_test.total_coupon = df_test.total_coupon.astype(float)
df_test.every_coupon_count = df_test.every_coupon_count.astype(float)


# In[16]:


#划分数据集
test = df_off.sample(frac=0.25, axis=0)
train = df_off.drop(np.array(test.index), axis=0)

valid = test[['User_id', 'Coupon_id']]
train.drop(['User_id', 'Coupon_id', 'Merchant_id'], axis=1, inplace=True)
test.drop(['User_id', 'Coupon_id', 'Merchant_id'], axis=1, inplace=True)


# In[17]:


test_x = test.drop('label', axis=1)
test_y = test[['label']]
train_x = train.drop('label', axis=1)
train_y = train[['label']]

valid['label'] = test_y


# In[18]:


test.label.value_counts()


# In[19]:


#构建模型
def check_model(train_x, train_y):
    classifier = XGBClassifier(n_estimators=1260,max_depth=8, min_child_weight=4, gamma=0.2, subsample=0.6, colsample_bytree=0.8, scale_pos_weight=20, reg_alpha=0, seed=0)
    model = Pipeline(steps=[('ss', StandardScaler()), ('en', classifier)])
    parameters = {'en__learning_rate': [0.01]}
    folder = StratifiedKFold(n_splits=3, shuffle=True)
    grid_search = GridSearchCV(model, parameters, cv=folder, n_jobs=-1, verbose=1)
    grid_search = grid_search.fit(train_x, train_y)
    return grid_search


# In[20]:


model = check_model(train_x, train_y)


# In[21]:


#验证
pred = model.predict_proba(test_x)
valid['pre_prob'] = pred[:, 1]


# In[22]:


#测试验证集的AUC
vg = valid.groupby('Coupon_id')
aucs = []
for i in vg:
    tmpdf = i[1]
    if len(tmpdf['label'].unique())!=2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pre_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))


# In[23]:


#预测
pred = model.predict_proba(df_test)
result['Probability'] = pred[:, 1]
result.sort_values(by=['Coupon_id','Probability'],inplace=True)
result.to_csv("test_predction.csv",index=None,header=None)

