import pandas as pd
import numpy as np
import time
import lightgbm as lgb

# 读历史曝光日志数据文件，加列名
# 大约1ww条
data = pd.read_csv('../testA/totalExposureLog.out', header=None)
data.columns = ['req_id', 'req_time', 'adStand_id',
                'user_id', 'adExp_id', 'scale', 'bid',
                'pctr', 'quality_ecpm', 'totalEcpm']


def stamp2ymd(stamp):
    t = time.localtime(stamp)
    ymd = time.strftime('%Y%m%d', t)
    return ymd


# 新增一个日期列，e.g.‘20190317’
data['ymd'] = data['req_time'].map(stamp2ymd)

# groupby（广告名，尺寸，出价），按日期计算曝光量
# 聚合之后，（广告名，尺寸，出价）这样日曝光数据有4kw条，50%+的日曝光都只有1
data_add_exp = data.groupby(['adExp_id', 'scale', 'bid'])['ymd'].value_counts().to_frame('dailyExp_count').reset_index()

# 读测试集，2w条
data_test = pd.read_table('../testA/test_sample.dat', header=None)
data_test.columns = ['sample_id', 'adExp_id', 'initime', 'scale', 'adField_id', 'goodtype', 'good_id',
                     'account_id', 'period', 'people', 'bid']
data_test['iniymd'] = data_test['initime'].map(stamp2ymd)

# 下面数据清洗的操作可能对模型影响很大
# 只挑选广告id在测试集里的曝光数据作为训练集， 也就是 曝光数据∩测试集
data_clean_test = data_test[['adExp_id', 'iniymd', 'adField_id', 'goodtype', 'account_id']].drop_duplicates(
    keep='first')
train_data = pd.merge(data_add_exp, data_clean_test, on='adExp_id')

# 给训练集加一个表示广告从创建开始过了几天的列
train_data['ago'] = pd.to_datetime(train_data['ymd']) - pd.to_datetime(train_data['iniymd'])
train_data['ago'] = train_data['ago'].map(lambda x: x / np.timedelta64(1 * 60 * 60 * 24, 's'))

# 训练集施工完毕（广告id，尺寸，商品域，商品类型，出价，已上线天数，日曝光量）
train_data_f = train_data[['adExp_id', 'scale', 'adField_id', 'goodtype', 'bid',
                           'ago', 'dailyExp_count']]
train_label = train_data_f['dailyExp_count']
train_data_fe = train_data_f.drop(columns='dailyExp_count')

# 测试集施工完毕（广告id，尺寸，商品域，商品类型，出价，已上线天数）
data_test0 = data_test[['adExp_id', 'scale', 'adField_id', 'goodtype', 'bid', 'iniymd']]
data_test0['ago'] = pd.to_datetime('20190320') - pd.to_datetime(data_test0['iniymd'])
data_test0['ago'] = data_test0['ago'].map(lambda x: x / np.timedelta64(1 * 60 * 60 * 24, 's'))
data_test0['ago'].mask(data_test0['ago'] <= 0, 1, inplace=True)
data_test_f = data_test0.drop(columns='iniymd')

# lgbm
trainset = lgb.Dataset(train_data_fe, label=train_label)
param = {
    'num_leaves': 150,
    'max_depth': 5,
    'learning_rate': .05,
    'max_bin': 200

}

num_round = 22
lgbm = lgb.train(param, trainset, num_round)

test_pre = lgbm.predict(data_test_f)

# 输出
sub = pd.DataFrame(test_pre)
sub4 = sub.applymap(lambda x: '%.4f' % x)
sub4.insert(0, 'sample_id', value=data_test['sample_id'])
sub4.to_csv('../submit/sub.csv', index=0, header=0)
exit()
## bid/10作为预测值
sub2 = pd.DataFrame(data_test0['bid'])
sub2res = sub2.applymap(lambda x: '%.4f' % (x / 10))
sub2res.insert(0, 'sample_id', value=data_test['sample_id'])
sub2res.to_csv('submission_bid_reduce10.csv', index=0, header=0)