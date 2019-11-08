import pandas as pd
import time
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
files_path = '../testA/'
df_operate = pd.read_csv(files_path + 'ad_operation.dat',
                               names=['ad_id', 'update_time', 'operator_type', 'change_word',
                                      'word_after_change_operator'],low_memory=False,sep='\t')

df_static = pd.read_csv(files_path + 'ad_static_feature.out',
                                     names=['ad_id', 'init_time', 'ad_account_id', 'goods_id', 'goods_type',
                                            'ad_industry_id', 'material_size'],sep='\t',low_memory=False)
print(len(df_operate))
df_operate= df_operate[df_operate.update_time==0]
print(len(df_operate))
exit()

df_state,df_bid,df_group,df_period = df_operate[df_operate.change_word==1],df_operate[df_operate.change_word==2],\
                                     df_operate[df_operate.change_word==3],df_operate[df_operate.change_word==4]

df_bid.rename(columns={'word_after_change_operator':'bid_1'},inplace=True)
df_group.rename(columns={'word_after_change_operator':'group'},inplace=True)
df_state.rename(columns={'word_after_change_operator':'state'},inplace=True)
df_period.rename(index=str,columns={'word_after_change_operator':'period'},inplace=True)


df = pd.merge(df_operate,df_bid[['ad_id','bid_1']],on='ad_id',how='left')
df = pd.merge(df,df_group[['ad_id','group']],on='ad_id',how='left')
df = pd.merge(df,df_state[['ad_id','state']],on='ad_id',how='left')
df = pd.merge(df,df_period[['ad_id','period']],on='ad_id',how='left')
df.period.fillna('-999',inplace=True)
df.state.fillna(1,inplace=True)
del df['word_after_change_operator']
del df['change_word']
# del df['operator_type']
df.drop_duplicates(inplace=True)

print(df)
# exit()
df_static = pd.merge(df_static,df,on='ad_id',how='left')
df_static.drop_duplicates(inplace=True)
print(df_static)