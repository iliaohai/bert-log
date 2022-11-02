'''
    使用transformers进行文本分类
    Date: 20220124
    Author: Hai Liao
    Version: 1.0
'''
import pandas as pd
import transformers as tfs
from torch import nn
from torch import optim
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from transformers import AdamW
from utils.logutil import Logger
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差

# 记录日志
filename = "../../OutPut/unilm/run_HDFS_by10%"
time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
filepath = filename + "_" + time_stamp + ".log"
log = Logger(filepath, level='debug')

class BertClassificationModel(nn.Module):
    def __init__(self):
        super(BertClassificationModel, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (tfs.BertForSequenceClassification, tfs.BertTokenizer, 'microsoft/unilm-base-cased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert = model_class.from_pretrained(pretrained_weights, num_labels=2)

    def forward(self, batch_sentences, batch_labels):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True, max_length=512,  padding='longest')  # tokenize、add special token、pad
        input_ids = torch.tensor(batch_tokenized['input_ids']).cuda()
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda()
        batch_labels = torch.tensor(batch_labels).cuda()
        bert_output = self.bert(input_ids, attention_mask=attention_mask, labels=batch_labels)
        return bert_output

'''
    1.加载数据集
'''
train_sets = pd.read_csv('../../Dataset/HDFS/data_instances2.csv', header=0)  # 训练集

# 数据集大小分割，分别取10%，20%，50%
# train_sets = train_sets[0: int(len(train_sets) * 0.1)]

x = train_sets['EventSequence'].values
y = train_sets['Label'].values

# 数据集分配
train_inputs, dev_inputs, train_targets, dev_targets = train_test_split(x, y, test_size=0.20, random_state=32)
#
# print(len(train_inputs))
# print(len(train_targets))
# print(len(dev_inputs))
# print(len(dev_targets))

'''
    2.训练数据集分批
'''
batch_size = 16
batch_count = int(len(train_inputs) / batch_size)
batch_train_inputs, batch_train_targets = [], []
for i in range(batch_count):
    j = i * batch_size
    batch_sentences = []
    while j < (i+1) * batch_size:
        # 每一列CLS
        # print("第", j, "列")
        # print(train_inputs[j])
        sens = train_inputs[j][1: len(train_inputs[j])-2]
        sens = sens.split(",")
        cls = ""
        for e in sens:
            e = e.replace(' ', '')
            e = e.replace("'", '')
            cls += e + " "
        cls = "[CLS]" + cls + "[SEB]"
        # print(cls)
        batch_sentences.append(cls)
        j = j + 1
    batch_train_inputs.append(batch_sentences)  # 带batch的cls
    batch_train_targets.append(train_targets[i*batch_size: (i+1)*batch_size])  # 带batch的label

'''
    3.训练模型
'''
epochs = 4
# lr = 0.01
print_every_batch = 1
bert_classifier_model = BertClassificationModel().cuda()
# optimizer = optim.SGD(bert_classifier_model.parameters(), lr=lr, momentum=0.9)
optimizer = AdamW(bert_classifier_model.parameters(), lr=2e-5)

# 记录最好结果
max_acc, max_f1, max_p, max_r, best_epoch = 0.0, 0.0, 0.0, 0.0, 0
for epoch in range(epochs):
    print_avg_loss = 0
    bert_classifier_model.train()
    for i in range(batch_count):
        # 正向传播
        inputs = batch_train_inputs[i]  # inputs中有64条训练数据
        labels = batch_train_targets[i]  # labels中有64条标签
        optimizer.zero_grad()
        outputs = bert_classifier_model(inputs, labels)
        loss = outputs[0]
        print_avg_loss += loss.item()

        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bert_classifier_model.parameters(), 1.0)
        optimizer.step()

        if i % print_every_batch == (print_every_batch - 1):
            log.logger.info("epoch:%d" % epoch)
            log.logger.info("Batch: %d, Loss: %.4f" % ((i + 1), print_avg_loss / print_every_batch))
            print_avg_loss = 0

    '''
        4.评价模型
    '''
    bert_classifier_model.eval()
    total = len(dev_targets)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for j in range(total):
            dev_input = dev_inputs[j]
            sens = dev_input[1: len(dev_input) - 2]
            sens = sens.split(",")
            cls = ""
            for e in sens:
                e = e.replace(' ', '')
                e = e.replace("'", '')
                cls += e + " "
            cls = "[CLS]" + cls + "[SEB]"
            outputs = bert_classifier_model([cls], dev_targets[j])  # 此处需要传数组，所以加一个[]
            loss = outputs[0]
            logits = outputs[1]
            y_pred.append(torch.argmax(logits, axis=1)[0].item())
            y_true.append(dev_targets[j])
    print(y_true)
    print(y_pred)
    f1score = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')  # 0.2222222222222222
    recall = recall_score(y_true, y_pred, average='macro')
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    log.logger.info("***************** Result ******************")
    log.logger.info("epoch:%d" % epoch)
    log.logger.info("Accuracy:%s" % acc)
    log.logger.info("F1:%s" % f1score)
    log.logger.info("precision:%s" % precision)
    log.logger.info("recall:%s" % recall)
    log.logger.info("mse:%s" % mse)
    log.logger.info("mae:%s" % mae)
    if (acc + f1score)/2 > (max_acc + max_f1) / 2:
        max_acc, max_f1, max_p, max_r, best_epoch = acc, f1score, precision, recall, epoch
    log.logger.info("max_accuracy:%s" % max_acc)
    log.logger.info("max_f1:%s" % max_f1)
    log.logger.info("max_p:%s" % max_p)
    log.logger.info("max_r:%s" % max_r)
    log.logger.info("best_epoch:%d" % best_epoch)
    log.logger.info("******************************************")

    # # 绘制roc图
    # Roc_figure.plot_ROC(y_true, y_pred, "(a) ROC based on HDFS dataset")

    # 导出预测结果
    # result = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    # result.to_csv('../../../Dataset/HDFS/result_BERT-Log.csv', index=False)
