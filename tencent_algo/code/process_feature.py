import pandas as pd
import time
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
files_path = '../testA/'

import datetime

def smape(pred, data):
    F = data.get_label()
    A = pred
    return 'smape',100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))),True


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

now_time = time.time()


df_total_log = pd.read_hdf(files_path + 'totalExposureLog_filter.h5',mode='a')
# df_total_log = df_total_log.loc[:1000000,:]
print(df_total_log)
print(time.time()-now_time)

df_operate = pd.read_csv(files_path + 'ad_operation.dat',
                               names=['ad_id', 'update_time', 'operator_type', 'change_word',
                                      'word_after_change_operator'],low_memory=False,sep='\t')

df_static = pd.read_csv(files_path + 'ad_static_feature.out',
                                     names=['ad_id', 'init_time', 'ad_account_id', 'goods_id', 'goods_type',
                                            'ad_industry_id', 'material_size'],sep='\t',low_memory=False)

df_test = pd.read_csv(files_path + 'Btest_sample_new.dat',names=['item_id', 'ad_id','init_time','material_size',
                                                              'ad_industry_id','goods_type','goods_id','ad_account_id',
                                                              'bid_period','group_orient','bid'],sep='\t',low_memory=False)

df_static = df_static[['ad_id', 'init_time', 'ad_account_id', 'goods_id', 'goods_type','ad_industry_id', ]]

df_static = df_static.loc[df_static.init_time.isnull()==False,:]

df_total_log['req_datetime'] = df_total_log['req_time'].map(stamp2datetime)
df_total_log['ymd'] = df_total_log['req_time'].map(stamp2ymd)

df_total_log = pd.merge(df_total_log,df_static,on='ad_id',how='left')

df_total_log['init_datetime'] = df_total_log['init_time'].map(stamp2datetime)
df_total_log['init_ymd'] = df_total_log['init_time'].map(stamp2ymd)
df_total_log = df_total_log.loc[df_total_log.init_ymd>'1970-01-01',:]

df_test['init_datetime'] = df_test['init_time'].map(stamp2datetime)
df_test['init_ymd'] = df_test['init_time'].map(stamp2ymd)

df_test['ymd'] = '2019-03-20'
df_total_log['part_ecpm'] = df_total_log['totalExpm']-df_total_log['quality_ecpm']

df_train = df_total_log.copy()



tmp = df_total_log.groupby(['ad_id','ymd'])['req_time'].agg({'dayExpm':'count'},as_index=False).reset_index()

df_train = pd.merge(df_total_log,tmp,on=['ad_id','ymd'] ,how='left')
print(df_train)




all_cols = ['material_size', 'bid', 'pctr', 'quality_ecpm','totalExpm','day']

# print(len(df))

for j,group in enumerate(['ad_id', 'goods_type', 'material_size','ad_account_id']):
    print(j + 1,group)
    for i,col in enumerate(['bid', 'pctr','quality_ecpm', 'totalExpm','part_ecpm']):
        print(col)
        cols_list = [group+'_'+col+'_mean',group+'_'+col+'_median',group+'_'+col+'_max',group+'_'+col+'_min']
        tmp = df_total_log.groupby([group,'ymd'])[col].agg({group+'_'+col+'_mean':'mean',group+'_'+col+'_max':'max',
            group+'_'+col+'_min':'min',group+'_'+col+'_median':'median'},as_index=False).reset_index()
        try:
            df_train = pd.merge(df_train, tmp, on=[group, 'ymd'], how='left')
            del tmp['ymd']
            # df_test = pd.merge(df_test,tmp.drop_duplicates(subset=[group],keep='last'),on=[group], how='left')
            df_test = pd.merge(df_test, tmp.groupby([group])[cols_list].max().reset_index(), on=[group], how='left')

            all_cols += cols_list
            # print(df_test.columns)
            print(len(df_test))
        except:
            print('group_by empty for '+ group+'_'+col)
            continue

        # all_cols.append(group+'_'+col+'_mean')
        # all_cols.append(group + '_' + col + '_median')
        # all_cols.append(group + '_' + col + '_max')
        # all_cols.append(group + '_' + col + '_min')
print('done!')

# id_list = ['ad_account_id', 'goods_type', 'material_size','ad_ask_id', 'ad_position_id', 'user_id']

id_list = ['ad_account_id', 'goods_id', 'goods_type','ad_industry_id',
           'ad_ask_id', 'ad_position_id', 'user_id']

for i,col in enumerate(['ad_account_id', 'goods_type', 'material_size']):
    print(i+1,col)
    tmp = df_total_log.groupby([col])['ad_id'].nunique().reset_index(name=col+'_ad_id_nunique')
    try:
        df_train = pd.merge(df_train, tmp, on=[col], how='left')
        # df_test = pd.merge(df_test, tmp.drop_duplicates(subset=[col], keep='last'), on=[col], how='left')
        df_test = pd.merge(df_test, tmp.groupby([col])[col+'_ad_id_nunique'].max().reset_index(), on=[col], how='left')
        all_cols.append(col + '_ad_id_nunique')
        print(len(df_test))

    except:
        print('nanana')
        continue



for i,col in enumerate(id_list):
    print(i+1,col)
    tmp = df_total_log.groupby(['ad_id','ymd'])[col].nunique().reset_index(name=col+'_nunique')
    try:
        df_train = pd.merge(df_train, tmp, on=['ad_id','ymd'], how='left')
        df_test = pd.merge(df_test, tmp.groupby(['ad_id'])[col+'_nunique'].max().reset_index(), on=['ad_id'], how='left')
        all_cols.append(col + '_nunique')
        print(len(df_test))

    except:
        print('nanana')
        continue


for i,col in enumerate(id_list):
    print(i+1,col)
    tmp = df_total_log.groupby(['ad_id','material_size'])[col].nunique().reset_index(name=col+'_material_size_nunique')
    try:
        df_train = pd.merge(df_train, tmp, on=['ad_id','material_size'], how='left')
        df_test = pd.merge(df_test, tmp.groupby(['ad_id'])[col+'_material_size_nunique'].max().reset_index(), on=['ad_id'], how='left')
        all_cols.append(col + '_material_size_nunique')
        print(len(df_test))

    except:
        print('nanana')
        continue


print('done!')

print(df_total_log)



df_train = df_train.drop_duplicates(subset=['ad_id','ymd','material_size'],keep='first')
df_train.sort_values('dayExpm',inplace=True)
print(df_train['dayExpm'])

df_train['month'] = df_train['ymd'].apply(lambda x:x[5:7]).astype(str)
df_train['day'] = df_train['ymd'].apply(lambda x:x[8:10]).astype(str)
df_train.sort_values('dayExpm',inplace=True)


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

print(df_train.columns)
print(all_cols)



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
        print(len(df_test))



df_train['goods_id'] = df_train['goods_id'].apply(trans_int)
df_train['ad_industry_id'] = df_train['ad_industry_id'].apply(trans_int)

df_test = pd.merge(df_test,df_train[['ad_id','material_size','pctr','quality_ecpm','totalExpm']].drop_duplicates
                (subset=['ad_id','material_size'],keep='last'),on=['ad_id','material_size'],how='left')

df_test['ago'] = 32
df_test['day'] = 21


for i in range(1,4):
    col_name = 'dayExpm_last_'+str(i)
    tmp1 = df_train.loc[(df_train['ago']==(31-i)),['ad_id','dayExpm']]
    tmp1.columns = ['ad_id',col_name]
    print(tmp1)
    tmp_list = list(set(tmp1.ad_id).intersection(set(df_test.ad_id)))
    df_test = pd.merge(df_test,tmp1.loc[(tmp1.ad_id.isin(tmp_list)),['ad_id',col_name]]
                       ,how='left',on=['ad_id']).reset_index(drop=True)

    all_cols.append(col_name)
    print(len(df_test))



print(all_cols)
all_cols += ['ad_id','goods_type','ad_account_id','goods_id','ad_industry_id']

print(df_train[['dayExpm','dayExpm_last_1','dayExpm_last_2','dayExpm_last_3']])


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


tmp1 = df_train.groupby(['ad_id'])['dayExpm'].agg({'dayExpm_mean':'mean','dayExpm_max':'max','dayExpm_median':'median','dayExpm_min':'min'}).reset_index()
df_test = df_test.merge(tmp1,how='left',on='ad_id')

all_cols.append('dayExpm_mean')


print(df_test)
print(df_test.columns)
all_cols.append('item_id')

tmp1 = df_train.loc[(df_train.ymd>='2019-02-22'),:].groupby(['ad_id'])['dayExpm'].agg({'dayExpm_recent':'mean'}).reset_index()
df_test = df_test.merge(tmp1,how='left',on='ad_id')

all_cols += ['dayExpm_min','dayExpm_max','dayExpm_median','dayExpm_recent']



df_train.to_csv(files_path + 'df_train.csv',index=False)
df_test[all_cols].to_csv(files_path+'df_testB.csv',index=False)


print(time.time()-now_time)
