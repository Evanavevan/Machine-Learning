
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import date


# In[2]:


df_off = pd.read_csv('ccf_offline_stage1_train.csv')
df_test = pd.read_csv('ccf_offline_stage1_test_revised.csv')


# In[ ]:


df_off.head(10)


# In[ ]:


df_test.head(10)


# ![image.png](attachment:image.png)

# In[ ]:


print("有优惠券：{}".format(df_off.Date_received.isnull().value_counts()[0]))
print("没有优惠券：{}".format(df_off.Date_received.isnull().value_counts()[1]))


# In[ ]:


print("购买商品：{}".format(df_off.Date.isnull().value_counts()[0]))
print("未购买商品：{}".format(df_off.Date.isnull().value_counts()[1]))


# In[ ]:


print("有优惠券，购买商品：{}".format(df_off[df_off.Date_received.notnull() & df_off.Date.notnull()].shape[0]))
print("有优惠券，未购买商品：{}".format(df_off[df_off.Date_received.notnull() & df_off.Date.isnull()].shape[0]))
print("没有优惠券，购买商品：{}".format(df_off[df_off.Date_received.isnull() & df_off.Date.notnull()].shape[0]))
print("没有优惠券，未购买商品：{}".format(df_off[df_off.Date_received.isnull() & df_off.Date.isnull()].shape[0]))


# In[3]:


#将年月日提取出来
def get_year_month_day(df):
    df['year'] = df.Date_received.astype(str).apply(lambda x:0 if x == 'nan' else int(x[0:4]) )
    df['month'] = df.Date_received.astype(str).apply(lambda x:0 if x == 'nan' else int(x[4:6]))
    df['day'] = df.Date_received.astype(str).apply(lambda x:0 if x == 'nan' else int(x[6:8]))
    return df

df_off = get_year_month_day(df_off)
df_test = get_year_month_day(df_test)


# ### 提取特征

# In[4]:


#每个顾客使用优惠券的频率:单独使用掉优惠券的概率,一个月内使用劵的概率
#领劵的次数，领劵的数量，领劵的时间，一个月内领劵的数量，领同样的券的数量前后时间等等
#统计用户一个月以来领到的券的总数
def get_all_coupon_count(df):
    t = df[df.Coupon_id.notnull()][['User_id']]
    t['user_receive_all_coupon_count'] = 1
    return t.groupby('User_id').agg('sum').reset_index()

def get_all_coupon_use_count(df):
    t = df[df.Coupon_id.notnull()&df.Date.notnull()][['User_id']]
    t['user_use_all_coupon_count'] = 1
    return t.groupby('User_id').agg('sum').reset_index()

def get_all_same_coupon_count(df):
    t = df[df.Coupon_id.notnull()][['User_id', 'Coupon_id']]
    t['user_receive_same_coupon_count'] = 1
    return t.groupby(['User_id', 'Coupon_id']).agg('sum').reset_index()

def get_all_same_coupon_use_count(df):
    t = df[df.Coupon_id.notnull()&df.Date.notnull()][['User_id', 'Coupon_id']]
    t['user_use_same_coupon_count'] = 1
    return t.groupby(['User_id', 'Coupon_id']).agg('sum').reset_index()

def get_month_all_coupon_count(df):
    t = df[df.Coupon_id.notnull()][['User_id', 'month']]
    t['this_month_user_receive_all_coupon_count'] = 1
    return t.groupby(['User_id', 'month']).agg('sum').reset_index()

def get_month_same_coupon_count(df):
    t = df[df.Coupon_id.notnull()][['User_id', 'Coupon_id',  'month']]
    t['this_month_user_receive_same_coupon_count'] = 1
    return t.groupby(['User_id', 'Coupon_id', 'month']).agg('sum').reset_index()

def fill(df):
    df.user_use_all_coupon_rate.fillna(0, inplace=True)
    df.user_use_same_coupon_rate.fillna(0, inplace=True)
    df.user_use_all_coupon_count.fillna(0, inplace=True)
    df.user_use_same_coupon_count.fillna(0, inplace=True)
    df.user_receive_all_coupon_count.fillna(0, inplace=True)
    df.user_receive_same_coupon_count.fillna(0, inplace=True)
    df.this_month_user_receive_all_coupon_count.fillna(0, inplace=True)
    df.this_month_user_receive_same_coupon_count.fillna(0, inplace=True)
    return df


# In[5]:


t1 = get_month_all_coupon_count(df_off)
t3 = get_month_same_coupon_count(df_off)
t5 = get_all_coupon_count(df_off)
t6 = get_all_coupon_use_count(df_off)
t7 = get_all_same_coupon_count(df_off)
t8 = get_all_same_coupon_use_count(df_off)

t9 = get_all_coupon_count(df_test)
t10 = get_all_same_coupon_count(df_test)
t11 = get_month_all_coupon_count(df_test)
t12 = get_month_same_coupon_count(df_test)

t13 = pd.merge(t5, t6, on = 'User_id')
t14 = pd.merge(t7, t8, on = ['User_id', 'Coupon_id'])
t = pd.merge(t13, t14, on='User_id')
df_off = pd.merge(df_off, t, on = ['User_id', 'Coupon_id'], how='left')
df_off['user_use_all_coupon_rate'] = df_off.user_use_all_coupon_count / df_off.user_receive_all_coupon_count
df_off['user_use_same_coupon_rate'] = df_off.user_use_same_coupon_count / df_off.user_receive_same_coupon_count
#df_off.drop(['user_use_all_coupon_count', 'user_use_same_coupon_count'], axis=1, inplace=True)

df_off = pd.merge(df_off, t1, on = ['User_id', 'month'], how='left')
df_off = pd.merge(df_off, t3, on = ['User_id', 'Coupon_id', 'month'], how='left')
df_off = fill(df_off)
df_off.head(10)


# In[6]:


t = pd.merge(t9, t10, on = 'User_id')
df_test = pd.merge(df_test, t, on = ['User_id', 'Coupon_id'], how='left')
t = df_off[['User_id', 'user_use_all_coupon_rate']]
df_test = pd.merge(df_test, t, on='User_id', how='left')
t = df_off[['User_id', 'user_use_same_coupon_rate', 'Coupon_id']]
df_test = pd.merge(df_test, t, on = ['User_id', 'Coupon_id'], how='left')
t = pd.merge(t11, t12, on=['User_id', 'month'])
df_test = pd.merge(df_test, t, on = ['User_id', 'Coupon_id', 'month'], how='left')
df_test['user_use_all_coupon_count'] = df_test.user_receive_all_coupon_count * df_test.user_use_all_coupon_rate
df_test['user_use_same_coupon_count'] = df_test.user_receive_same_coupon_count * df_test.user_use_same_coupon_rate
df_test = fill(df_test)
df_test.head(20)


# In[7]:


t = df_test.user_use_all_coupon_count
df_test.drop('user_use_all_coupon_count', axis=1, inplace=True)
df_test.insert(10, 'user_use_all_coupon_count', t)
df_test.head()


# In[8]:


t = df_test.user_use_same_coupon_count
df_test.drop('user_use_same_coupon_count', axis=1, inplace=True)
df_test.insert(12, 'user_use_same_coupon_count', t)
df_test.head()


# In[9]:


#商品的种类，商品的营销额，商户的距离等等
mi = df_off[['Merchant_id', 'Coupon_id', 'Date']]

t1 = mi[mi.Date.notnull()][['Merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('Merchant_id').agg('sum').reset_index()
df_off = pd.merge(df_off, t1, on='Merchant_id', how='left')

t2 = mi[mi.Date.notnull()&mi.Coupon_id.notnull()][['Merchant_id']]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('Merchant_id').agg('sum').reset_index()
df_off = pd.merge(df_off, t2, on='Merchant_id', how='left')

t3 = mi[mi.Coupon_id.notnull()][['Merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('Merchant_id').agg('sum').reset_index()
df_off = pd.merge(df_off, t3, on='Merchant_id', how='left')

#df_off['coupon_rate'] = df_off.sales_use_coupon / df_off.total_sales
#df_off['tranfer_rate'] = df_off.sales_use_coupon / df_off.total_coupon
df_off.drop(['total_sales', 'sales_use_coupon'], axis=1, inplace=True)
df_off.head(10)


# In[10]:


t3 = df_test[['Merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('Merchant_id').agg('sum').reset_index()
df_test = pd.merge(df_test, t3, on='Merchant_id', how='left')


# In[11]:


df_test.head()


# In[12]:


#门店距离
print(df_off.Distance.unique())
print(df_test.Distance.unique())


# In[13]:


#填充空值
df_off['distance'] = df_off['Distance']
df_off.distance.fillna(-1, inplace=True)
print(df_off.distance.unique())
df_test['distance'] = df_test['Distance']
df_test.distance.fillna(-1, inplace=True)
print(df_test.distance.unique())


# In[14]:


def process_distance(t):
    t.distance = t.distance.astype(int)
    t.replace(-1, np.nan, inplace=True)

    t1 = t.groupby('Merchant_id').agg('min').reset_index()
    t1.replace(np.nan, -1, inplace=True)
    t1.rename(columns={'distance':'mechant_min_distance'}, inplace=True)

    t2 = t.groupby('Merchant_id').agg('max').reset_index()
    t2.replace(np.nan, -1, inplace=True)
    t2.rename(columns={'distance':'mechant_max_distance'}, inplace=True)

    t3 = t.groupby('Merchant_id').agg('mean').reset_index()
    t3.replace(np.nan, -1, inplace=True)
    t3.rename(columns={'distance':'mechant_mean_distance'}, inplace=True)

    t4 = t.groupby('Merchant_id').agg('min').reset_index()
    t4.replace(np.nan, -1, inplace=True)
    t4.rename(columns={'distance':'mechant_median_distance'}, inplace=True)
    return t1, t2, t3, t4

#合并
def join(df, t):
    df = pd.merge(df, t, on='Merchant_id', how='left')
    return df


# In[15]:


d = df_off[['Merchant_id', 'distance', 'Date', 'Date_received']]
t = df_off[df_off.Date.notnull()&df_off.Date_received.notnull()][['Merchant_id', 'distance']]
t1, t2, t3, t4 = process_distance(t)
df_off = join(df_off, t1)
df_off = join(df_off, t2)
df_off = join(df_off, t3)
df_off = join(df_off, t4)
df_off.head(20)


# In[16]:


t = df_test[['Merchant_id', 'distance']]
t1, t2, t3, t4 = process_distance(t)
df_test = join(df_test, t1)
df_test = join(df_test, t2)
df_test = join(df_test, t3)
df_test = join(df_test, t4)
df_test.head(20)


# In[17]:


df_off.drop('Distance', axis=1, inplace=True)
df_test.drop('Distance', axis=1, inplace=True)


# In[18]:


#优惠券的种类，优惠率之类的
#优惠券的种类以及所花出去的频率：不限时间的，每个月内的
ci = df_off[['Coupon_id', 'Date', 'month']]

t1 = ci[ci.Coupon_id.notnull()][['Coupon_id']]
t1['every_coupon_count'] = 1
t1 = t1.groupby('Coupon_id').agg('sum').reset_index()
df_off = pd.merge(df_off, t1, on='Coupon_id', how='left')

#t2 = ci[ci.Coupon_id.notnull()&ci.Date.notnull()][['Coupon_id']]
#t2['every_coupon_use_count'] = 1
#t2 = t2.groupby('Coupon_id').agg('sum').reset_index()
#df_off = pd.merge(df_off, t2, on='Coupon_id', how='left')
#df_off['every_coupon_use_rate'] = df_off.every_coupon_use_count / df_off.every_coupon_count

#t3 = ci[ci.Coupon_id.notnull()][['Coupon_id', 'month']]
#t3['every_month_coupon_count'] = 1
#t3 = t3.groupby(['Coupon_id', 'month']).agg('sum').reset_index()
#df_off = pd.merge(df_off, t3, on=['Coupon_id', 'month'], how='left')

#t4 = ci[ci.Coupon_id.notnull()&ci.Date.notnull()][['Coupon_id', 'month']]
#t4['every_month_coupon_use_count'] = 1
#t4 = t4.groupby(['Coupon_id', 'month']).agg('sum').reset_index()
#df_off = pd.merge(df_off, t4, on=['Coupon_id', 'month'], how='left')
#df_off['every_month_coupon_use_rate'] = df_off.every_month_coupon_use_count / df_off.every_month_coupon_count

df_off.every_coupon_count.fillna(0, inplace=True)
#df_off.every_coupon_use_count.fillna(0, inplace=True)
#df_off.every_coupon_use_rate.fillna(0, inplace=True)
#df_off.every_month_coupon_count.fillna(0, inplace=True)
#df_off.every_month_coupon_use_count.fillna(0, inplace=True)
#df_off.every_month_coupon_use_rate.fillna(0, inplace=True)


# In[19]:


df_off.head(20)


# In[20]:


t1 = df_test[['Coupon_id']]
t1['every_coupon_count'] = 1
t1 = t1.groupby('Coupon_id').agg('sum').reset_index()
df_test = pd.merge(df_test, t1, on='Coupon_id', how='left')
#t2 = df_off[['Coupon_id', 'every_coupon_use_rate']]
#df_test = pd.merge(df_test, t2, on='Coupon_id', how='left')
df_test.head(20)


# In[21]:


#优惠率
#分为打折，满减和不优惠三种模式
#影响人的购买程度
df_off.Discount_rate.unique()


# In[22]:


#获取大折扣的类型：0代表不打折扣，1代表采用折扣比率，2表示满减的形式
def get_discount_type(s):
    s = str(s)
    if s == 'null':
        return 0.0
    elif ':' in s:
        return 1.0
    else:
        return 2.0

#计算满减的折扣率，用比例来展现会比较直观，因为有券购买商品的人数太少，用满减的门槛来定义显得有些过于刻意
def cal_discount_rate(s):
    s = str(s)
    if s == 'null':
        return 1.0
    else:
        s = s.split(':')
        if len(s) == 1:
            return float(s[0])
        else:
            return 1.0 - float(s[1]) / float(s[0])
    
#获取满减的门槛
def get_discount_man(s):
    s = str(s)
    if ':' in s:
        s = s.split(':')
        return float(s[0])
    else:
        return 0.0
    
#获取满减的钱数：
def get_discount_jian(s):
    s = str(s)
    if ':' in s:
        s = s.split(':')
        return float(s[1])
    else:
        return 0.0    

    #定义一个总函数
def process_discount(df):
    df['discount_type'] = df['Discount_rate'].apply(get_discount_type)
    df['discount_rate'] = df['Discount_rate'].apply(cal_discount_rate)
    df['discount_man'] = df['Discount_rate'].apply(get_discount_man)
    df['discount_jian'] = df['Discount_rate'].apply(get_discount_jian)
    df.discount_rate.fillna(0.0, inplace=True)
    return df


# In[23]:


df_off = process_discount(df_off)
df_test = process_discount(df_test)


# In[24]:


df_off.drop('Discount_rate', axis=1, inplace=True)
df_off.head(20)


# In[25]:


df_test.drop('Discount_rate', axis=1, inplace=True)
df_test.head(10)


# In[26]:


#时间特征，包括节假日，工作日与休息日
df_off.Date_received.unique()


# In[27]:


df_test.Date_received.unique()


# In[28]:


#获取工作日与休息日
def get_weekday(s):
    if s =='nan':
        return 0.0
    else:
        return date(int(s[0:4]), int(s[4:6]), int(s[6:8])).weekday() + 1    #weekday返回的数为0~6，代表一周

df_off['weekday'] = df_off.Date_received.astype(str).apply(get_weekday)
df_test['weekday'] = df_test.Date_received.astype(str).apply(get_weekday)


# In[29]:


df_off.weekday.value_counts()


# In[30]:


df_test.weekday.value_counts()


# In[31]:


#将一周内几天分类，周一周五周六周日为一个种类，记为1，剩余的记为0
def get_weekday_type(df):
    df['weekday_type'] = df['weekday'].apply(lambda x:1 if x in [1, 5, 6, 7] else 0)
    return df

df_off = get_weekday_type(df_off)
df_test = get_weekday_type(df_test)
df_off.head(10)


# In[32]:


df_test.head(10)


# In[33]:


#独热编码
def one_hot(df, s):
    a = pd.get_dummies(df.weekday, prefix = s)
    df = pd.concat([df, a], axis = 1)
    return df

df_off = one_hot(df_off, 'weekday')
df_test = one_hot(df_test, 'weekday')

df_off.drop('weekday', axis=1, inplace=True)
df_off.head(10)


# In[34]:


df_test.drop('weekday', axis=1, inplace=True)
df_test.rename(columns={'weekday_1':'weekday_1.0',  'weekday_2':'weekday_2.0', 'weekday_3':'weekday_3.0', 'weekday_4':'weekday_4.0', 'weekday_5':'weekday_5.0', 'weekday_6':'weekday_6.0', 'weekday_7':'weekday_7.0'}, inplace=True)
df_test.head(10)


# In[35]:


#处理标签
def get_label(s):
    s = s.split(':')
    if s[0] == 'nan':
        return 0
    elif s[1] == 'nan':
        return -1
    elif (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8]))-date(int(s[1][0:4]), int(s[1][4:6]), int(s[1][6:8]))).days<=15:
        return 1
    else:
        return 0
df_off['label'] = (df_off.Date.astype(str) + ':' + df_off.Date_received.astype(str)).apply(get_label)
df_off.head(10)


# In[36]:


#删除无用信息
df_off.drop(['year', 'month', 'day', 'Date'], axis=1, inplace=True)
df_test.drop(['year', 'month', 'day'], axis=1, inplace=True)
df_off.head(10)


# In[44]:


df_off.isnull().sum()


# In[40]:


#填充空值
#df_off.isnull().sum()
df_off.Coupon_id.fillna(0, inplace=True)
#df_off.total_sales.fillna(0, inplace=True)
df_off.total_coupon.fillna(0, inplace=True)
#df_off.sales_use_coupon.fillna(0, inplace=True)
#df_off.coupon_rate.fillna(0, inplace=True)
#df_off.tranfer_rate.fillna(0, inplace=True)
df_off.mechant_min_distance.fillna(-1, inplace=True)
df_off.mechant_max_distance.fillna(-1, inplace=True)
df_off.mechant_mean_distance.fillna(-1, inplace=True)
df_off.mechant_median_distance.fillna(-1, inplace=True)
df_off.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# In[39]:


df_off = df_off.drop(np.array(df_off[df_off.label==-1].index), axis=0)
df_off.drop('weekday_0.0', axis=1, inplace=True)


# In[41]:


#去重
df_off.drop_duplicates(inplace=True)
df_test.drop_duplicates(inplace=True)


# In[ ]:


df_off.shape


# In[ ]:


df_test.shape


# In[42]:


df_off.index = range(len(df_off))


# In[43]:


df_test.index = range(len(df_test))


# In[ ]:


df_test


# In[45]:


#保存特征数据集
df_off.to_csv('train_feature.csv', index=None)
df_test.to_csv('test_feature.csv', index=None)

