import math
from sklearn import linear_model
import pandas as pd
import time
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)

files_path = '../testA/'
# files_path = r"C:\Users\hzp\Desktop\tencent_data\"

now_time = time.time()
# # read ad_operator.dat,time=tiny
# df_ad_operator = pd.read_table(files_path + 'ad_operation.dat',
#                                names=['ad_id', 'new_change_time', 'operator_type', 'change_word',
#                                       'word_after_change_operator'],low_memory=False)
# print(type(df_ad_operator))
# print(df_ad_operator.info())
# print(df_ad_operator.head(10))
#
# # read ad_static_feature, time = tiny
# df_ad_static_feature = pd.read_table(files_path + 'ad_static_feature.out',
#                                      names=['ad_id', 'new_time', 'ad_account_id', 'goods_id', 'goods_type',
#                                             'ad_industry_id', 'material_size'],low_memory=False)
# print(df_ad_static_feature.info())
# print(df_ad_static_feature.head(4))
#
# # read total_exposure_log , time = 67s
df_total_exposure_log = pd.read_table(files_path + 'totalExposureLog.out',
                                      names=['ad_ask_id', 'ad_ask_time', 'ad_position_id', 'user_id', 'ad_id',
                                             'material_size', 'bid_price', 'pctr',
                                             'quality_ecpm', 'totalExpm'],low_memory=False)
#
# print(df_total_exposure_log.info())
# print(df_total_exposure_log.head(10))
#
# # read user_data , time = 30s
# df_user_data = pd.read_table(files_path + 'user_data', names=['user_id', 'age', 'gender', 'area', 'marriage_status',
#                                                               'education', 'consuption_ability', 'device',
#                                                               'work_status', 'connection_type', 'behavior'],low_memory=False)
#
# print(df_user_data.info())
# print(df_user_data.head(10))

df_test_sample = pd.read_table(files_path + 'test_sample.dat',names=['item_id', 'ad_id','new_time','material_size',
                                                              'ad_industry_id','goods_type','goods_id','ad_account_id',
                                                              'bid_period','group_orient','bid_price'],low_memory=False )


df_test_sample['new_time'] = df_test_sample['new_time'].apply(lambda x:time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(x)))
print(df_test_sample.info())
print(df_test_sample.head(10))
print(time.time() - now_time)

