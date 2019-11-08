import datetime
now = datetime.datetime.now()
print(now)
now = '20190512'
# d1 = datetime.datetime(now)
# d2 = datetime.datetime(now)

d1 = datetime.datetime.strptime('2019-02-17','%Y-%m-%d')
d2 = datetime.datetime.strptime('2019-03-20','%Y-%m-%d')
print((d2-d1).days)
# now = datetime.datetime.strptime(now,'%Y%m%d')
# date = now - datetime.timedelta(days = 1)
# date = date.strftime('%Y%m%d')
# print(date)
#
# from itertools import combinations
# id_cols = ['ad_id','ad_industry_id','ad_account_id','goods_id','goods_type']
# id_cols_1 = [list(i) for i in combinations(id_cols,1)]
# id_cols_2 = [list(i) for i in combinations(id_cols,2)]
# id_cols_3 = [list(i) for i in combinations(id_cols,3)]
# id_cols_3 = []
# id_cols_all = id_cols_1+id_cols_2+id_cols_3
# id_cols_all = id_cols_1
# print(id_cols_all)


print(list(df_train.columns))

stat_cols = ['pctr','totalExpm','quality_ecpm','bid']

all_cols = ['pctr','totalExpm','quality_ecpm',]


# all_cols+=id_cols
# mms = MinMaxScaler(feature_range=(0, 1))
# df_train[all_cols] = mms.fit_transform(df_train[all_cols])
# exit()
# for col in stat_cols:
#     for combine in id_cols_all:
#         combine_str = '_'.join(combine)
#         all_cols.append(col+'_'+combine_str+'_max')
#         all_cols.append(col + '_' + combine_str + '_min')
#         all_cols.append(col + '_' + combine_str + '_mean')
#         print(combine_str)
#         tmp = df_train.groupby(combine, as_index=False)[col].agg({col+'_'+combine_str+'_max': 'max',
#                                                                    col+'_'+combine_str+'_min': 'min',
#                                                                     col+'_'+combine_str+'_mean': 'mean'
#                                                                     })
#         df_train = df_train.merge(tmp, on=combine, how='left')
#         df_test = df_test.merge(tmp,on=combine, how='left')