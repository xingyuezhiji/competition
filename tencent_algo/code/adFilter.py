import pandas as pd
import time
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
files_path = '../testA/'


now_time = time.time()
# df_total_log = pd.read_csv(files_path + 'totalExposureLog.out',
#                                       names=['ad_ask_id', 'req_time', 'ad_position_id', 'user_id', 'ad_id',
#                                              'material_size', 'bid', 'pctr',
#                                              'quality_ecpm', 'totalExpm'],low_memory=False,sep='\t',nrows=500000)

# df_total_log.to_hdf(files_path + 'totalExposureLog.h5','df',mode='w',format='table',data_columns=True)
df_total_log = pd.read_hdf(files_path + 'totalExposureLog.h5',mode='a')
print(df_total_log)
print(time.time()-now_time)

df_operate = pd.read_csv(files_path + 'ad_operation.dat',
                               names=['ad_id', 'update_time', 'operator_type', 'change_word',
                                      'word_after_change_operator'],low_memory=False,sep='\t')

df_static = pd.read_csv(files_path + 'ad_static_feature.out',
                                     names=['ad_id', 'init_time', 'ad_account_id', 'goods_id', 'goods_type',
                                            'ad_industry_id', 'material_size'],sep='\t',low_memory=False)

df_test_sample = pd.read_csv(files_path + 'update_Btest_sample.dat',names=['item_id', 'ad_id','init_time','material_size',
                                                              'ad_industry_id','goods_type','goods_id','ad_account_id',
                                                              'bid_period','group_orient','bid'],sep='\t',low_memory=False)

df_static = df_static[['ad_id', 'init_time', 'ad_account_id', 'goods_id', 'goods_type','ad_industry_id', ]]

# df_static = df_static.loc[df_static.init_time.isnull()==False,:]
df_operate= df_operate[df_operate.update_time==0]
df_static = df_static.loc[df_static.ad_id.isin(list(df_operate.ad_id))|(df_static.ad_id.isin(list(df_test_sample.ad_id))),:].reset_index(drop=True)
df_total_log = df_total_log.loc[df_total_log.ad_id.isin(list(df_operate.ad_id))|(df_total_log.ad_id.isin(list(df_test_sample.ad_id))),:].reset_index(drop=True)
# df_total_log = df_total_log.loc[df_total_log.req_time.isnull()==False,:].reset_index(drop=True)
df_total_log = df_total_log.loc[(df_total_log.quality_ecpm>=0)&(df_total_log.quality_ecpm<18000)
                                &(df_total_log.totalExpm<1e6)&(df_total_log.bid<2e7)&(df_total_log.pctr<=1000),:].reset_index(drop=True)

print(df_total_log)
df_total_log.to_hdf(files_path + 'totalExposureLog_filter.h5','df',mode='w',format='table',data_columns=True)
print(time.time()-now_time)