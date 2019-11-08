import math
from sklearn import linear_model
import pandas as pd
import time
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)

files_path = '../testA/'
# files_path = r"C:\Users\hzp\Desktop\tencent_data\"

now_time = time.time()

df_total_exposure_log = pd.read_table(files_path + 'totalExposureLog.out',
                                      names=['ad_ask_id', 'ad_ask_time', 'ad_position_id', 'user_id', 'ad_id',
                                             'material_size', 'bid_1', 'pctr',
                                             'quality_ecpm', 'totalExpm'],low_memory=False,nrows=500000)

# df_total_exposure_log['rate'] = df_total_exposure_log['bid']/df_total_exposure_log['totalExpm']
# print(df_total_exposure_log['rate'].describe())
# print(df_total_exposure_log[['bid','totalExpm']].head(100))
# print(df_total_exposure_log.head(200))
# exit()

df_test_sample = pd.read_table(files_path + 'test_sample.dat',names=['item_id', 'ad_id','new_time','material_size',
                                                              'ad_industry_id','goods_type','goods_id','ad_account_id',
                                                              'bid_period','group_orient','bid'],low_memory=False )
print('read data done!')
df_temp = df_total_exposure_log.groupby('ad_id')['quality_ecpm'].\
    agg({'quality_ecpm_mean':'mean','quality_ecpm_max':'max','quality_ecpm_min':'min'})

# print(df_temp)

df_temp = df_total_exposure_log.groupby('ad_id')

# for name,group in df_temp:
#     print(name)
#     print(group)
# exit()
# df_test_sample = pd.merge(df_test_sample,df_temp,on='ad_id',how='left')
# print(1)
# df_temp = df_total_exposure_log.groupby('ad_id')['pctr'].agg({'pctr_mean':'mean'})
# df_test_sample = pd.merge(df_test_sample,df_temp,on='ad_id',how='left')
# print(2)
# df_temp = df_total_exposure_log.groupby('ad_id')['totalExpm'].agg({'totalExpm_mean':'mean'})
# df_test_sample = pd.merge(df_test_sample,df_temp,on='ad_id',how='left')
# print(3)
df_test = df_test_sample.copy()
df_test_sample = pd.merge(df_test_sample,df_total_exposure_log.drop_duplicates(subset=['ad_id']),on='ad_id',how='left')
# print(df_total_exposure_log)
# exit()
# df_test_sample = df_test_sample.loc[df_test.item_id,:]
df_test_sample.fillna(0,inplace=True)
df_test_sample['pred_totalExpm'] = df_test_sample['bid']*df_test_sample['pctr']+df_test_sample['quality_ecpm']
print(len(df_test_sample),len(df_test_sample.loc[df_test_sample.pred_totalExpm==0,'pred_totalExpm']))
df_test_sample.loc[df_test_sample.pred_totalExpm==0,'pred_totalExpm'] = \
    df_test_sample.loc[df_test_sample.pred_totalExpm==0,'bid'].values/10
# df_test_sample[['item_id','pred_totalExpm']].to_csv('../submit/sub_rule.csv',header=None,index=None)

# df_test.set_index('item_id')[['ad_id', 'bid']].groupby('ad_id')['bid'].apply(lambda row: pd.Series(dict(zip(row.index, row.rank()/ 6)))).round(4).to_csv('../submit/submission.csv', header=None)
# df_test_sample['new_time'] = df_test_sample['new_time'].apply(lambda x:time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(x)))

temp = df_test.set_index('item_id')[['ad_id', 'bid']].groupby('ad_id')['bid']
print(temp)
for a,x in temp:
    print(a)
    print(x.rank())