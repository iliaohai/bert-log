#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer.models import LR
from loglizer import dataloader, preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

struct_log = '../data/HDFS/HDFS.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv'  # The anomaly label file

if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
    #                                                             label_file=label_file,
    #                                                             window='session',
    #                                                             train_ratio=0.8,
    #                                                             split_type='uniform')

    '''
           1.加载数据集
       '''
    train_sets = pd.read_csv('../../../Dataset/HDFS/data_instances2.csv', header=0)  # 训练集
    # 数据集大小分割，分别取10%，20%，50%
    train_sets = train_sets[0: int(len(train_sets) * 0.01)]
    x = train_sets['EventSequence'].values
    y = train_sets['Label'].values
    # 数据集分配
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=32)

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    model = LR()
    model.fit(x_train, y_train)

    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)

    print('mse:')
    mse = model.mse(x_test, y_test)
    print(mse)
