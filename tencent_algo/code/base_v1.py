import pandas as pd
import time
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
files_path = '../testA/'
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

def stamp2ymd(stamp):
    try:
        t = time.localtime(stamp)
        ymd = time.strftime('%Y-%m-%d',t)
    except:
        ymd='2019-03-20'
    return ymd

def trans_int(item):
    try:
        item = int(item)
    except:
        item = -1
    return item

def add_ago(item):
    try:
        d1 = datetime.datetime.strptime(item['first_ymd'], '%Y-%m-%d')
        d2 = datetime.datetime.strptime(item['ymd'], '%Y-%m-%d')
        return (d2-d1).days
    except:
        return 0



df_train = pd.read_csv(files_path+'df_train.csv')

print(df_train)
df_train.sort_values('ymd',inplace=True)
print(df_train)
print(df_train['dayExpm'].describe())


df_train = df_train.loc[(df_train['init_time'].isnull()==False)|(df_train['ad_industry_id'].isnull()==False)
                        |(df_train['goods_type'].isnull()==False)|(df_train['goods_id'].isnull()==False),:]

print(len(df_train))
df_train = df_train.drop_duplicates(subset=['ymd','ad_id'])
print(len(df_train))
df_train = df_train[df_train['dayExpm']<1000]
print(df_train['dayExpm'].describe())

# df_total_log = pd.read_hdf(files_path + 'totalExposureLog.h5',mode='a')
df_test = pd.read_csv(files_path+'df_testB.csv')

# df_test = pd.read_csv(files_path + 'test_sample.dat',names=['item_id', 'ad_id','init_time','material_size',
#                                                               'ad_industry_id','goods_type','goods_id','ad_account_id',
#                                                               'bid_period','group_orient','bid'],sep='\t',low_memory=False)


all_cols = ['material_size', 'bid', 'pctr', 'quality_ecpm','totalExpm',
            'bid_mean', 'bid_max', 'bid_median', 'bid_min','pctr_mean', 'pctr_max','pctr_median', 'pctr_min',
            'quality_ecpm_mean', 'quality_ecpm_max', 'quality_ecpm_median', 'quality_ecpm_min',
            'part_ecpm_mean', 'part_ecpm_max', 'part_ecpm_median', 'part_ecpm_min',
            'totalExpm_mean','totalExpm_max', 'totalExpm_median','totalExpm_min',
            'ad_account_id_nunique', 'goods_id_nunique','goods_type_nunique', 'ad_industry_id_nunique',
            'material_size_nunique','ad_ask_id_nunique', 'ad_position_id_nunique', 'user_id_nunique',
             'day']




df_train['first_ymd'] = '2019-02-16'
df_train['ago'] = df_train.apply(add_ago,axis=1)

for j in range(1,4):
    col_name = 'dayExpm_last_' + str(j)
    # all_cols.append(col_name)
    print(col_name)
    for day in range(4,32):
        tmp1 = df_train.loc[(df_train['ago']==day),['ad_id','dayExpm']]
        tmp2 = df_train.loc[(df_train['ago']==(day-j)),['ad_id','dayExpm']]
        tmp_list = list(set(tmp1.ad_id).intersection(set(tmp2.ad_id)))
        df_train.loc[(df_train.ad_id.isin(tmp_list))&(df_train.ago==day),col_name] = tmp2.loc[(tmp2.ad_id.isin(tmp_list)),'dayExpm'].values
        # print(df_train.loc[(df_train.day==10),'day1'])

# df_train = df_train.loc[df_train['ago']>=4,:]


merge_cols = all_cols.copy()
merge_cols.remove('bid')
merge_cols.append('ad_id')
# merge_cols.remove('init_day')

# all_cols.remove('ad_ask_id_nunique')


df_train['goods_id'] = df_train['goods_id'].apply(trans_int)
df_train['ad_industry_id'] = df_train['ad_industry_id'].apply(trans_int)

df_test = pd.merge(df_test,df_train[merge_cols].drop_duplicates(subset=['ad_id','material_size'],keep='last'),
                   on=['ad_id','material_size'],how='left')


df_test['ago'] = 32




for i in range(1,4):
    col_name = 'dayExpm_last_'+str(i)
    tmp1 = df_train.loc[(df_train['ago']==(31-i)),['ad_id','dayExpm']]
    tmp1.columns = ['ad_id',col_name]
    print(tmp1)
    tmp_list = list(set(tmp1.ad_id).intersection(set(df_test.ad_id)))
    df_test = pd.merge(df_test,tmp1.loc[(tmp1.ad_id.isin(tmp_list)),['ad_id',col_name]]
                       ,how='left',on=['ad_id']).reset_index(drop=True)

    all_cols.append(col_name)




print(all_cols)
all_cols += ['ad_id','goods_type','ad_account_id','goods_id','ad_industry_id']
# print()
# print(df_test)
print(df_train[['dayExpm','dayExpm_last_1','dayExpm_last_2','dayExpm_last_3']])

# exit()

df_train['init_month'] = df_train['init_ymd'].astype(str).apply(lambda x:x[5:7])
df_train['init_day'] = df_train['init_ymd'].astype(str).apply(lambda x:x[8:10])

df_test['init_month'] = df_test['init_ymd'].astype(str).apply(lambda x:x[5:7])
df_test['init_day'] = df_test['init_ymd'].astype(str).apply(lambda x:x[8:10])



df_train_copy = df_train.copy()
for i in range(0,32):
    print(i)
    tmp = df_train.loc[df_train.ago!=i,:].groupby(['ad_id'])['dayExpm'].agg({'dayExpm_mean':'mean'}).reset_index()
    df_temp = pd.merge(df_train_copy,tmp,how='left',on='ad_id').reset_index(drop=True)
    df_train.loc[(df_train.ago == i), 'dayExpm_mean'] = df_temp.loc[(df_temp.ago == i), 'dayExpm_mean'].values

df_train = df_train.loc[df_train['ago']>=4,:]
df_valid = df_train[df_train.ymd=='2019-03-19']
df_train = df_train[(df_train.ymd<'2019-03-19')]


# tmp = df_train.loc[(df_train.ymd>='2019-03-01'),:].groupby(['ad_id'])['dayExpm'].agg({'dayExpm_mean':'mean'}).reset_index()


# df_train = df_train.merge(tmp,how='left',on='ad_id')
# df_valid = df_valid.merge(tmp,how='left',on='ad_id')
tmp1 = df_train.groupby(['ad_id'])['dayExpm'].agg({'dayExpm_mean':'mean'}).reset_index()
df_test = df_test.merge(tmp1,how='left',on='ad_id')

all_cols.append('dayExpm_mean')

tmp1 = df_train.loc[(df_train.ymd>='2019-02-22'),:].groupby(['ad_id'])['dayExpm'].agg({'dayExpm_recent':'mean'}).reset_index()
df_test = df_test.merge(tmp1,how='left',on='ad_id')



# df_train['dayExpm'] = df_train['dayExpm'].apply(np.log1p)
# df_valid['dayExpm'] = df_valid['dayExpm'].apply(np.log1p)
print(df_train)
print(df_valid)
# print(df_test)

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from datetime import datetime
lgb_params = {
        "learning_rate": 0.02,
        # "lambda_l1": 0.1,
        # "lambda_l2": 0.2,
        "max_depth": 7,
        "num_leaves": 63,
        "objective": "regression",
        "verbose": -1,
        'feature_fraction': 1,
        # "min_split_gain": 0.1,
        "boosting_type": "gbdt",
        "subsample": 1,
        "min_data_in_leaf": 50,
        'max_bin':124,
        'metric':('l1', 'smape')
    }
train_data = df_train
test_data = df_test
valid_data = df_valid

print(df_test)

# exit()

train_data = train_data[all_cols+['dayExpm']]
test_data = test_data[all_cols]
X = train_data[all_cols]
y = train_data['dayExpm'].values
X_test = test_data[all_cols]
X_valid = valid_data[all_cols]
y_valid = valid_data['dayExpm'].values

lgb_train = lgb.Dataset(X, y)
lgb_evals = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

evals_result = {}
gbm = lgb.train(lgb_params,
                lgb_train,
                num_boost_round=1500,
                valid_sets=[lgb_train, lgb_evals],
                valid_names=['train', 'valid'],
                early_stopping_rounds=100,
                verbose_eval=20,
                evals_result=evals_result,
                )



ax = lgb.plot_metric(evals_result, metric='l1')#metric的值与之前的params里面的值对应
plt.show()

print('画特征重要性排序...')
ax = lgb.plot_importance(gbm, max_num_features=30)#max_features表示最多展示出前10个重要性特征，可以自行设置
plt.show()

print('Plot 3th tree...')  # 画出决策树，其中的第三颗
ax = lgb.plot_tree(gbm, tree_index=3, figsize=(20, 8), show_info=['split_gain'])
plt.show()

# print('导出决策树的pdf图像到本地')#这里需要安装graphviz应用程序和python安装包
# graph = lgb.create_tree_digraph(gbm, tree_index=3, name='Tree3')
# graph.render(view=True)




y_valid_pred = gbm.predict(X_valid)
# y_valid_pred = np.expm1(y_valid_pred)
# y_valid = np.expm1((y_valid))
mae = mean_absolute_error(y_valid,y_valid_pred)

print('valid mae: ',mae)

y_pred = gbm.predict(X_test)
# y_pred = np.expm1(y_pred)
print(y_pred)
print(y_pred.mean(),y_pred.min(),y_pred.max())

ad_id_list = list(set(df_train.ad_id).intersection(set(df_test.ad_id)))


sub = df_test[['item_id','ad_id','bid']]
# sub1 = df_test[['item_id','ad_id','bid']]
# sub1['pred'] = np.round(y_pred,4)
sub['pred'] = np.round(y_pred,4)

sub.loc[sub.ad_id.isin(ad_id_list),'pred'] = np.round(df_test.loc[df_test.ad_id.isin(ad_id_list),'dayExpm_mean'].values,4)
# sub.loc[sub.ad_id.isin(ad_id_list),'pred'] = np.round(df_test.loc[df_test.ad_id.isin(ad_id_list),'dayExpm_last_1'].values,4)


sub['pred'] = sub['pred'].apply(lambda x:1 if x<1 else x)
# sub1['pred'] = sub1['pred'].apply(lambda x:1 if x<1 else x)
sub['pred'] = sub['pred'] +sub['bid']/10000
# sub1['pred'] = sub1['pred'] +sub1['bid']/10000
sub1 = sub.copy()
sub1.sort_values('ad_id',inplace=True)
print(sub1[['ad_id','bid','pred']])
print(len(ad_id_list))
print(sub1.describe())
print(sub[['item_id','pred']].count())
sub[['item_id','pred']].to_csv('../submit/submit.csv',header=None,index=None)
