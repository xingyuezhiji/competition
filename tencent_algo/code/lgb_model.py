import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)


def smape_score(pred, data):
    labels = data.get_label()
    # pred = np.argmax(pred.reshape(12, -1), axis=0)      # lgb的predict输出为各类型概率值
    score = 1-(np.mean(np.abs(pred-labels)*2/(pred+labels)))/2
    return 'smape_score', score*0.4, True

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
        'metric':('l1', 'mape')
    }

df_train = pd.read_csv('../testA/df_train.csv')
df_test = pd.read_csv('../testA/df_testB.csv')

df1 =  pd.read_csv('../testA/ad_operation.csv')
df2 = pd.read_csv('../testA/ad_operation_target.csv')
df3 = pd.read_csv('../testA/Btest_sample_new.csv')
df4 = pd.read_csv('../testA/Btest_sample_new_target.csv')
df3.rename(columns={ df3.columns[0]: "item_id" }, inplace=True)
df4.rename(columns={ df4.columns[0]: "item_id" }, inplace=True)
print(df3)

print(len(df_train),len(df_test))
df_train = pd.merge(df_train,df1,how='left',on='ad_id')
df_train = pd.merge(df_train,df2,how='left',on='ad_id')
df_test = pd.merge(df_test,df3,how='left',on='item_id')
df_test = pd.merge(df_test,df4,how='left',on='item_id')

print(len(df_train),len(df_test))
print(df_train)
# exit()

print(df_test.columns)
#调阈值
df_train = df_train[df_train.dayExpm<800]

# df_train['ad_ask_id_nunique'].fillna(df_train['ad_ask_id_nunique'].mean(),inplace=True)
# df_train['dayExpm'] = df_train['dayExpm'].apply(np.log)

df_valid = df_train[df_train.ymd=='2019-03-19']
# df_train = df_train[(df_train.ymd<'2019-03-18')&(df_train.ymd>'2019-02-22')]
df_train = df_train[(df_train.ymd!='2019-03-19')]
all_cols = list(df_test.columns)



# # df_val = df_train[df_train.ymd=='2019-03-19']
# print(df_valid[['ad_id','dayExpm','dayExpm_last_1','dayExpm_last_2','dayExpm_last_3']].describe())
# # df_test.drop_duplicates('ad_id',inplace=True)
# print(df_test[['ad_id','dayExpm_last_1','dayExpm_last_2','dayExpm_last_3']].describe())
# # print(df_valid)
# exit()
print(all_cols)
# all_cols.remove('ad_id')
print(df_valid.dayExpm.describe())
# exit()
#


all_cols.remove('frame_detatil')
# all_cols.remove('ad_ask_id_nunique')

try:
    all_cols.remove('item_id')
except:
    pass
train_data = df_train
test_data = df_test
valid_data = df_valid

print(df_test)
all_cols.remove('dayExpm_recent')
all_cols.remove('dayExpm_min')
all_cols.remove('dayExpm_max')
all_cols.remove('dayExpm_median')

#调特征，下面有特征筛选
all_cols = ['ad_ask_id_nunique', 'user_id_nunique', 'ad_id_pctr_mean', 'ad_position_id_nunique',
            'ad_id_pctr_median', 'ad_id_quality_ecpm_mean', 'ad_position_id_material_size_nunique',
            'ad_id_bid_min', 'ad_id_quality_ecpm_median', 'pctr', 'ad_id_pctr_min', 'area',
            'ad_id_pctr_max', 'dayExpm_last_2', 'goods_type_quality_ecpm_max', 'dayExpm_last_1',
            'dayExpm_mean', 'ad_id', 'ad_account_id_bid_max', 'material_size_pctr_median',
            'material_size_bid_max', 'ad_id_totalExpm_mean', 'totalExpm', 'ad_industry_id',
            'ad_id_bid_max', 'dayExpm_last_3', 'ad_id_totalExpm_median', 'ad_account_id_bid_mean',
            'material_size_totalExpm_max', 'user_id_material_size_nunique', 'quality_ecpm',
            'ad_account_id_pctr_mean', 'ad_account_id', 'goods_type_totalExpm_max',
            'ad_ask_id_material_size_nunique', 'material_size_totalExpm_median', 'goods_type_bid_mean',
            'goods_type_bid_max', 'ad_id_part_ecpm_mean', 'ad_account_id_bid_min', 'goods_type_pctr_median',
            'ad_id_quality_ecpm_max', 'ad_account_id_bid_median', 'ad_id_bid_mean', 'material_size_part_ecpm_mean',
            'material_size', 'material_size_pctr_mean', 'ad_account_id_pctr_median', 'goods_type_part_ecpm_mean',
            'age', 'material_size_pctr_max', 'material_size_totalExpm_mean', 'goods_type_pctr_max',
            'ad_account_id_quality_ecpm_mean', 'ad_account_id_part_ecpm_median', 'ad_id_part_ecpm_min', 'bid_1']

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
                num_boost_round=5500,
                valid_sets=[lgb_train, lgb_evals],
                valid_names=['train', 'valid'],
                early_stopping_rounds=100,
                verbose_eval=20,
                evals_result=evals_result,
                feval=smape_score,
                # obj=smape
                )


imp = pd.DataFrame()
imp['feat'] = all_cols
imp['importance'] = gbm.feature_importance()
imp.sort_values('importance',inplace=True,ascending=False)

#特征筛选
imp = imp.loc[imp.importance>0.01*imp.importance.max(),:]
print(imp)
print(list(imp['feat']))

# lgb.plot_metric(evals_result, metric='l1')#metric的值与之前的params里面的值对应
# plt.show()
# #
# print('画特征重要性排序...')
# lgb.plot_importance(gbm, max_num_features=30)#max_features表示最多展示出前10个重要性特征，可以自行设置
# plt.show()
#
# print('Plot 3th tree...')  # 画出决策树，其中的第三颗
# lgb.plot_tree(gbm, tree_index=3, figsize=(20, 8), show_info=['split_gain'])
# plt.show()
#
# print('导出决策树的pdf图像到本地')#这里需要安装graphviz应用程序和python安装包
# graph = lgb.create_tree_digraph(gbm, tree_index=3, name='Tree3')
# graph.render(view=True)




y_valid_pred = gbm.predict(X_valid)
# y_valid_pred = np.exp(y_valid_pred)
# y_valid = np.exp((y_valid))
mae = mean_absolute_error(y_valid,y_valid_pred)

print('valid mae: ',mae)

y_pred = gbm.predict(X_test)
# y_pred = np.exp(y_pred)
print(y_pred)
print(y_pred.mean(),y_pred.min(),y_pred.max())

ad_id_list = list(set(df_train.ad_id).intersection(set(df_test.ad_id)))


sub = df_test[['item_id','ad_id','bid']]
# sub1 = df_test[['item_id','ad_id','bid']]
# sub1['pred'] = np.round(y_pred,4)
sub['pred'] = np.round(y_pred,4)
print(sub.describe())
sub.loc[sub.ad_id.isin(ad_id_list),'pred'] = np.round(df_test.loc[df_test.ad_id.isin(ad_id_list),'dayExpm_mean'].values,4)

sub.loc[sub.ad_id.isin(ad_id_list)==False,:].plot.scatter('ad_id','pred')
plt.show()
sub.loc[sub.ad_id.isin(ad_id_list)==True,:].plot.scatter('ad_id','pred')
plt.show()
print(sub.loc[sub.ad_id.isin(ad_id_list)==False,'pred'].describe())
exit()
# sub.loc[sub.ad_id.isin(ad_id_list),'pred'] = np.round(df_test.loc[df_test.ad_id.isin(ad_id_list),'dayExpm_last_1'].values,4)
a = sub['pred'].median()
print('median:',a)
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
sub[['item_id','pred']].to_csv('../submit/submission.csv',header=None,index=None)
print('median:',a)