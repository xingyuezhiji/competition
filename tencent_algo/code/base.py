import math
from sklearn import linear_model
import pandas as pd
import time
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
# from model import xDeepFM_MTL
files_path = '../testA/'
import matplotlib.pyplot as plt
import seaborn as sns

def smape(pred, data):
    F = data.get_label()
    A = pred
    return 'smape',100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))),True

# def f1_score_weighted(pred, data):
#     labels = data.get_label()
#     pred = np.argmax(pred.reshape(12, -1), axis=0)      # lgb的predict输出为各类型概率值
#     score = f1_score(y_true=labels, y_pred=pred, average='weighted')
#     return 'f1_score_weighted', score, True

def precesss_orient(data):
    #字典对应pandas每一列
    df = pd.DataFrame(data['group_orient'].tolist())
    data = pd.concat([data, df], axis=1)

    return data

def to_dict(s):
    try:
        #转字典
        a = dict((l.split(':') for l in s.split(',')))
    except:
        a={}
    return a

def stamp2ymd(stamp):
    try:
        t = time.localtime(stamp)
        ymd = time.strftime('%Y-%m-%d',t)
    except:
        ymd='2019-03-20'
    return ymd

def stamp2datetime(stamp):
    try:
        t = time.localtime(stamp)
        date = time.strftime('%Y-%m-%d %H:%M:%S',t)
    except:
        date='2019-03-20 00:00:00'
    return date

now_time = time.time()
# df_total_log = pd.read_csv(files_path + 'totalExposureLog.out',
#                                       names=['ad_ask_id', 'req_time', 'ad_position_id', 'user_id', 'ad_id',
#                                              'material_size', 'bid', 'pctr',
#                                              'quality_ecpm', 'totalExpm'],low_memory=False,sep='\t',nrows=500000)

# df_total_log.to_hdf(files_path + 'totalExposureLog.h5','df',mode='w',format='table',data_columns=True)
df_total_log = pd.read_hdf(files_path + 'totalExposureLog.h5',mode='a')
print(df_total_log)
print(time.time()-now_time)

# sns.scatterplot(x='ad_id', y='quality_ecpm', data=df_total_log)
# plt.show()
#
# sns.scatterplot(x='ad_id', y='totalExpm', data=df_total_log)
# plt.show()
#
# sns.scatterplot(x='ad_id', y='bid', data=df_total_log)
# plt.show()
#
# sns.scatterplot(x='ad_id', y='pctr', data=df_total_log)
# plt.show()
# exit()

df_operate = pd.read_csv(files_path + 'ad_operation.dat',
                               names=['ad_id', 'new_change_time', 'operator_type', 'change_word',
                                      'word_after_change_operator'],low_memory=False,sep='\t')
#
# df_user_data = pd.read_csv(files_path + 'user_data', names=['user_id', 'age', 'gender', 'area', 'status',
#                                                               'education', 'consuptionAbility', 'device',
#                                                               'work', 'connectionType', 'behavior'],low_memory=False,sep='\t',nrows=500000)
#
df_static = pd.read_csv(files_path + 'ad_static_feature.out',
                                     names=['ad_id', 'init_time', 'ad_account_id', 'goods_id', 'goods_type',
                                            'ad_industry_id', 'material_size'],sep='\t',low_memory=False)

df_test_sample = pd.read_csv(files_path + 'Btest_sample_new.dat',names=['item_id', 'ad_id','init_time','material_size',
                                                              'ad_industry_id','goods_type','goods_id','ad_account_id',
                                                              'bid_period','group_orient','bid'],sep='\t',low_memory=False)

df_static = df_static[['ad_id', 'init_time', 'ad_account_id', 'goods_id', 'goods_type','ad_industry_id', ]]

df_static = df_static.loc[df_static.init_time.isnull()==False,:]

df_static = df_static.loc[df_static.ad_id.isin(list(df_operate.ad_id)),:].reset_index(drop=True)
df_total_log = df_total_log.loc[df_total_log.ad_id.isin(list(df_operate.ad_id)),:].reset_index(drop=True)
df_total_log = df_total_log.loc[df_total_log.req_time.isnull()==False,:].reset_index(drop=True)
df_total_log = df_total_log.loc[(df_total_log.quality_ecpm>=0)&(df_total_log.quality_ecpm<18000)
                                &(df_total_log.totalExpm<1e6)&(df_total_log.bid<2e7)&(df_total_log.pctr<=1000),:].reset_index(drop=True)

print(df_total_log)
# exit()
# df_static['init_datetime'] = pd.to_datetime(df_static['init_time'],unit='s')
# df_static['init_ymd'] = df_static['init_datetime'].dt.strftime('%Y-%m-%d')
# df_total_log['req_datetime'] = pd.to_datetime(df_total_log['req_time'],unit='s')
# df_total_log['ymd'] = df_total_log['req_datetime'].dt.strftime('%Y-%m-%d')

df_total_log['req_datetime'] = df_total_log['req_time'].map(stamp2datetime)
df_total_log['ymd'] = df_total_log['req_time'].map(stamp2ymd)

# df_total_log = df_total_log.loc[df_total_log['ymd']>='2019-03-01',:]
df_total_log = pd.merge(df_total_log,df_static,on='ad_id',how='left')
# df_static['init_datetime'] = df_static['init_time'].map(stamp2datetime)
# df_static['init_ymd'] = df_static['init_time'].map(stamp2ymd)
# df_static = df_static.loc[df_static.init_ymd>'1970-01-01',:]
df_total_log['init_datetime'] = df_total_log['init_time'].map(stamp2datetime)
df_total_log['init_ymd'] = df_total_log['init_time'].map(stamp2ymd)
df_total_log = df_total_log.loc[df_total_log.init_ymd>'1970-01-01',:]



df_test_sample['init_datetime'] = df_test_sample['init_time'].map(stamp2datetime)
df_test_sample['init_ymd'] = df_test_sample['init_time'].map(stamp2ymd)

df_test_sample['ymd'] = '2019-03-20'

df_total_log['part_ecpm'] = df_total_log['totalExpm']-df_total_log['quality_ecpm']

for i,col in enumerate(['bid', 'pctr','quality_ecpm', 'totalExpm','part_ecpm']):
    print(i+1)
    tmp = df_total_log.groupby(['ad_id','ymd'])[col].agg({col+'_mean':'mean',col+'_max':'max',col+'_min':'min',col+'_median':'median'},as_index=False).reset_index()
    df_total_log = pd.merge(df_total_log, tmp, on=['ad_id', 'ymd'], how='left')

print('done!')


id_list = ['ad_account_id', 'goods_id', 'goods_type','ad_industry_id', 'material_size',
           'ad_ask_id', 'ad_position_id', 'user_id']

for i,col in enumerate(id_list):
    print(i+1)
    tmp = df_total_log.groupby(['ad_id','ymd'])[col].nunique().reset_index(name=col+'_nunique')
    df_total_log = pd.merge(df_total_log, tmp, on=['ad_id','ymd'], how='left')


print('done!')

print(df_total_log)
tmp = df_total_log.groupby(['ad_id','ymd'])['totalExpm'].agg({'dayExpm':'count'},as_index=False).reset_index()

data_add_exp = pd.merge(df_total_log,tmp,on=['ad_id','ymd'] ,how='left')
print(data_add_exp)
data_add_exp = data_add_exp.drop_duplicates(subset=['ad_id','ymd'],keep='first')


data_add_exp['dayExpm'] = data_add_exp['dayExpm'].fillna(0)
data_add_exp.sort_values('dayExpm',inplace=True)
# data_add_exp.reset_index(inplace=True)
print(data_add_exp['dayExpm'])
# exit()
df_train = data_add_exp
# data_clean_test = df_test_sample[['ad_id','init_datetime','init_ymd','ad_industry_id','goods_type','ad_account_id']].drop_duplicates(keep='first')
# df_train = pd.merge(data_add_exp,df_static,on='ad_id', how='left',left_index=True,right_index=True, suffixes=('', '_y'))
df_train['dayExpm'].fillna(0,inplace=True)
df_train['ymd'] = df_train['ymd'].astype(str)
df_train['month'] = df_train['ymd'].apply(lambda x:x[5:7]).astype(str)
df_train['day'] = df_train['ymd'].apply(lambda x:x[8:10]).astype(str)
df_train.sort_values('dayExpm',inplace=True)
# df_train = df_train.loc[df_train['ad_industry_id'].isnull()==False,:].reset_index()
print(df_train)
# df_train = df_train.loc[df_train['ymd']>='2019-03-01',:]
df_train.to_csv(files_path + 'df_train.csv',index=False)
df_test_sample.to_csv(files_path+'df_testB.csv',index=False)

print(time.time()-now_time)
exit()








