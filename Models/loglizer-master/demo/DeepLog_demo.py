#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
from loglizer import dataloader
from loglizer.models import DeepLog
from loglizer.preprocessing import Vectorizer, Iterator
from sklearn.model_selection import train_test_split
import pandas as pd

batch_size = 32
hidden_size = 32
num_directions = 2
topk = 5
train_ratio = 0.8
window_size = 10
epoches = 4
num_workers = 2
device = 0 

struct_log = '../data/HDFS/HDFS.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv'  # The anomaly label file

if __name__ == '__main__':
    (x_train, window_y_train, y_train), (x_test, window_y_test, y_test) = dataloader.load_HDFS(
                                                struct_log,
                                                label_file=label_file,
                                                window='session',
                                                window_size=window_size,
                                                train_ratio=train_ratio,
                                                split_type='uniform')
    radio = 0.01
    x_train = x_train[0: int(len(x_train) * radio)]
    window_y_train = window_y_train[0: int(len(window_y_train) * radio)]
    y_train = y_train[0: int(len(y_train) * radio)]
    x_test = x_test[0: int(len(x_test) * radio)]
    window_y_test = window_y_test[0: int(len(window_y_test) * radio)]
    y_test = y_test[0: int(len(y_test) * radio)]

    # '''
    #        1.加载数据集
    #    '''
    # train_sets = pd.read_csv('../../../Dataset/HDFS/data_instances2.csv', header=0)  # 训练集
    # # 数据集大小分割，分别取10%，20%，50%
    # train_sets = train_sets[0: int(len(train_sets) * 0.1)]
    # x = train_sets['EventSequence'].values
    # y = train_sets['Label'].values
    # # 数据集分配
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=32)
    # window_y_train = y_train
    # window_y_test = y_test

    feature_extractor = Vectorizer()
    train_dataset = feature_extractor.fit_transform(x_train, window_y_train, y_train)
    test_dataset = feature_extractor.transform(x_test, window_y_test, y_test)

    train_loader = Iterator(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers).iter
    test_loader = Iterator(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers).iter

    model = DeepLog(num_labels=feature_extractor.num_labels, hidden_size=hidden_size, num_directions=num_directions, topk=topk, device=device)
    model.fit(train_loader, epoches)

    print('Train validation:')
    metrics = model.evaluate(train_loader)

    print('Test validation:')
    metrics = model.evaluate(test_loader)
