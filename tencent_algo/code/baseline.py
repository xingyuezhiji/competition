import math
from sklearn import linear_model
import pandas as pd
import time
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
from model import xDeepFM_MTL
files_path = '../testA/'

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

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
        ymd = time.strftime('%Y%m%d',t)
    except:
        ymd='20190320'
    return ymd

now_time = time.time()
df_total_log = pd.read_csv(files_path + 'totalExposureLog.out',
                                      names=['ad_ask_id', 'req_time', 'ad_position_id', 'user_id', 'ad_id',
                                             'material_size', 'bid', 'pctr',
                                             'quality_ecpm', 'totalExpm'],low_memory=False,sep='\t',nrows=500000)

df_operate = pd.read_csv(files_path + 'ad_operation.dat',
                               names=['ad_id', 'new_change_time', 'operator_type', 'change_word',
                                      'word_after_change_operator'],low_memory=False,sep='\t')
#
df_user_data = pd.read_csv(files_path + 'user_data', names=['user_id', 'age', 'gender', 'area', 'status',
                                                              'education', 'consuptionAbility', 'device',
                                                              'work', 'connectionType', 'behavior'],low_memory=False,sep='\t',nrows=500000)
#
df_static = pd.read_csv(files_path + 'ad_static_feature.out',
                                     names=['ad_id', 'init_time', 'ad_account_id', 'goods_id', 'goods_type',
                                            'ad_industry_id', 'material_size'],sep='\t',low_memory=False)

df_test_sample = pd.read_csv(files_path + 'test_sample.dat',names=['item_id', 'ad_id','init_time','material_size',
                                                              'ad_industry_id','goods_type','goods_id','ad_account_id',
                                                              'bid_period','group_orient','bid'],sep='\t',low_memory=False)

df_static = df_static[['ad_id', 'init_time', 'ad_account_id', 'goods_id', 'goods_type','ad_industry_id', ]]
df_static['init_ymd'] = df_static['init_time'].map(stamp2ymd)
df_total_log['ymd'] = df_total_log['req_time'].map(stamp2ymd)
df_test_sample['init_ymd'] = df_test_sample['init_time'].map(stamp2ymd)
df_test_sample['ymd'] = '20190320'
data_add_exp = df_total_log.groupby(['ad_id','ymd'])['ad_id'].agg({'dayExpm':'count'}).reset_index()

# for j in data_add_exp:
#     print(j)
# exit()
# print(data_add_exp)
# exit()
# del df_total_log['ymd']
data_add_exp = pd.merge(data_add_exp,df_total_log,on=['ad_id','ymd'],how='left', left_index=True, right_index=True, suffixes=('', '_y'))
print(data_add_exp)
# exit()
data_clean_test = df_test_sample[['ad_id','init_ymd','ad_industry_id','goods_type','ad_account_id']].drop_duplicates(keep='first')
df_train = pd.merge(data_add_exp,df_static,on='ad_id', left_index=True, right_index=True, suffixes=('', '_y'))
print(df_train)

df_train['ago'] = pd.to_datetime(df_train['ymd']) - pd.to_datetime(df_train['init_ymd'])
df_train['ago'] = df_train['ago'].map(lambda x:x/np.timedelta64(1*60*60*24,'s'))

df_test_sample['ago'] = pd.to_datetime(df_test_sample['ymd']) - pd.to_datetime(df_test_sample['init_ymd'])
df_test_sample['ago'] = df_test_sample['ago'].map(lambda x:x/np.timedelta64(1*60*60*24,'s'))


user_attr_list = list(df_user_data.columns)
for attr in user_attr_list:
    try:
        df_user_data[attr] = df_user_data[attr].str.replace(',','|')
    except:
        pass

df_test_sample['group_orient'] = df_test_sample['group_orient'].str.replace('|',';').str.replace(',','|')\
                    .str.replace(';',',').apply(to_dict)

df_test_sample = precesss_orient(df_test_sample)
# df_train = pd.merge(df_train,df_user_data.drop_duplicates(subset=['user_id']),on='user_id',how='left', left_index=True, right_index=True, suffixes=('', '_y'))
df_train = pd.merge(df_train,df_operate.drop_duplicates(subset=['ad_id']),on='ad_id',how='left', left_index=True, right_index=True, suffixes=('', '_y'))
df_train = pd.merge(df_train,df_static.drop_duplicates(subset=['ad_id']),on='ad_id',how='left', left_index=True, right_index=True, suffixes=('', '_y'))


df_test_sample = pd.merge(df_test_sample,df_total_log,on=['ad_id','material_size','bid'],how='left', left_index=True, right_index=True, suffixes=('', '_y'))
df_test_sample = pd.merge(df_test_sample,df_operate.drop_duplicates(subset=['ad_id']),on='ad_id',how='left', left_index=True, right_index=True, suffixes=('', '_y'))

print(df_train)
# print(df_test_sample.columns,len(df_test_sample))
print(list(set(df_test_sample.columns).intersection(set(df_train.columns))))
# df_train.fillna(1,inplace=True)
# df_test_sample.fillna(1,inplace=True)

# all_data = pd.concat([df_train,df_test_sample], ignore_index=True,sort=False)

import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from deepctr.models import DeepFM,xDeepFM,NFFM,PNN,NFM
from deepctr.utils  import VarLenFeat, SingleFeat

key2index = {}
def split(x):
    key_ans = x.split('|')
    for key in key_ans:
        if key not in key2index:
            # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
            key2index[key] = len(key2index) + 1
    return list(map(lambda x: key2index[x], key_ans))
a = ['work', 'ad_id', 'education', 'connectionType', 'goods_type',
     'ad_industry_id', 'new_time', 'ad_account_id', 'consuptionAbility',
     'age', 'word_after_change_operator', 'goods_id', 'device', 'operator_type',
     'material_size', 'change_word', 'status', 'behavior', 'new_change_time',
     'gender', 'area']
sparse_features = ['goods_type', 'ad_id', 'goods_id',  'ad_account_id','ad_industry_id', 'ymd', 'init_ymd']
dense_features =['bid','pctr','quality_ecpm','totalExpm','material_size']

df_train[sparse_features] = df_train[sparse_features].fillna(0)
df_train[dense_features] = df_train[dense_features].fillna(0)

mms = MinMaxScaler(feature_range=(0, 1))
df_train[dense_features] = mms.fit_transform(df_train[dense_features])


for feat in sparse_features:
    try:
        df_train[feat] = df_train[feat].apply(int).astype(str)
    except:
        print(feat)
        print(type(df_train.loc[1,feat]))
        sparse_features.remove(feat)
# sparse_features.remove('gender')
print(sparse_features)

print(df_train[sparse_features])
target = ['dayExpm']
print(df_train[sparse_features+target+dense_features])


# 1.Label Encoding for sparse features,and process sequence features
for feat in sparse_features:
    lbe = LabelEncoder()
    df_train[feat] = lbe.fit_transform(df_train[feat])
# preprocess the sequence feature
multi_values_cols = ['status', 'work',  'device','education','connectionType','age',
                     'consuptionAbility', 'behavior','area', 'operator_type',]
multi_values_input = []
sequence_feature = []
# for col in multi_values_cols:
#     key2index = {}
#     df_train[col] = df_train[col].apply(str)
#     # all_data[col] = all_data[col].apply(str)
#     lst = list(map(split, df_train[col].values))
#     lst_all = list(map(split, df_train[col].values))
#     lst_length = np.array(list(map(len, lst_all)))
#     max_len = max(lst_length)
#     # Notice : padding=`post`
#     lst = pad_sequences(lst, maxlen=max_len, padding='post')
#     multi_values_input += [lst]
#     # 2.count #unique features for each sparse field and generate feature config for sequence feature
#     sequence_feature += [VarLenFeat(col, len(key2index) + 1, max_len, 'mean')]
#     # Notice : value 0 is for padding for sequence input feature

sparse_feat_list = [SingleFeat(feat, df_train[feat].nunique())
                        for feat in sparse_features]

dense_feat_list = [SingleFeat(feat, 0)
                          for feat in dense_features]
# 3.generate input data for model
sparse_input = [df_train[feat.name].values for feat in sparse_feat_list]
dense_input = [df_train[feat.name].values for feat in dense_feat_list]


model_input = sparse_input + dense_input + multi_values_input
print(model_input)
# print(model_input.shape)
# 4.Define Model,compile and train
model = DeepFM({"sparse": sparse_feat_list,"dense": dense_feat_list,"sequence": sequence_feature
                },final_activation='linear', embedding_size=8,
           use_fm=False, hidden_size=(64, 64))

model.compile("adam", "mape", metrics=['mape'],)
history = model.fit(model_input, df_train[target].values,
                    batch_size=2048, epochs=200, verbose=2, validation_split=0.2,)
pred = model.predict(model_input)
print(pred)
print(smape(df_train[target].values,pred))







