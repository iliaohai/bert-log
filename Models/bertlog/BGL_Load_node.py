import os
import pandas as pd
import numpy as np

para = {"window_size": 0.5, "step_size": 0.2, "structured_file": "data/BGL.log_structured2.csv", "BGL_sequence": 'BGL_sequence.csv'}

def load_BGL():

    structured_file = para["structured_file"]
    # load data
    bgl_structured = pd.read_csv(structured_file)
    # convert to data time format
    bgl_structured["Time"] = pd.to_datetime(bgl_structured["Time"], format= "%Y-%m-%d-%H.%M.%S.%f")
    # calculate the time interval since the start time
    bgl_structured["Seconds_since"] = (bgl_structured['Time']-bgl_structured['Time'][0]).dt.total_seconds().astype(int)
    # get the label for each log("-" is normal, else are abnormal label)
    bgl_structured['Label'] = (bgl_structured['Label'] != '-').astype(int)
    # 获取节点id
    # bgl_structured['Node'] =
    return bgl_structured


def bgl_sampling(bgl_structured):

    label_data,time_data,event_mapping_data, nodes = bgl_structured['Label'].values, bgl_structured['Seconds_since'].values,bgl_structured['EventId'].values, bgl_structured['Node'].values
    log_size = len(label_data)
    # split into sliding window
    start_time = time_data[0]
    start_index = 0
    end_index = 0
    start_end_index_list = []
    # get the first start, end index, end time
    for cur_time in time_data:
        if cur_time < start_time + para["window_size"]*3600:
            end_index += 1
            end_time = cur_time
        else:
            start_end_pair = tuple((start_index, end_index))
            start_end_index_list.append(start_end_pair)
            break
    while end_index < log_size:
        start_time = start_time + para["step_size"]*3600
        end_time = end_time + para["step_size"]*3600
        for i in range(start_index,end_index):
            if time_data[i] < start_time:
                i+=1
            else:
                break

        for j in range(end_index, log_size):
            if time_data[j] < end_time:
                j+=1
            else:
                break
        start_index = i
        end_index = j
        start_end_pair = tuple((start_index, end_index))
        start_end_index_list.append(start_end_pair)
    # start_end_index_list is the  window divided by window_size and step_size,
    # the front is the sequence number of the beginning of the window,
    # and the end is the sequence number of the end of the window
    inst_number = len(start_end_index_list)
    print('there are %d instances (sliding windows) in this dataset'%inst_number)

    # get all the log indexs in each time window by ranging from start_index to end_index
    expanded_indexes_list = [[] for i in range(inst_number)]
    expanded_event_list = [[] for i in range(inst_number)]
    expanded_node_list = [[] for i in range(inst_number)]
    expanded_label_list = [[] for i in range(inst_number)]

    for i in range(inst_number):
        expanded_node_list.append(i)
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        for l in range(start_index, end_index):
            expanded_indexes_list[i].append(l)  # 一个时间窗口下的索引数组
            expanded_event_list[i].append(event_mapping_data[l])  # 一个时间窗口下的事件数组
            expanded_node_list[i].append(nodes[l])  # 一个时间窗口下的节点数组
            expanded_label_list[i].append(label_data[l])  # 一个时间窗口下的标签数组

    '''
        对时间窗口下的数据按照节点进行重新组合
        author: liaohai
    '''
    end_nodes = []
    end_events = []
    end_labels = []

    big_nodes = []
    big_events = []
    big_labels = []

    for h in range(inst_number):
        # print("每个时间窗口下的节点数组")
        node_window = expanded_node_list[h]   # 一个窗口下的节点数组
        enent_window = expanded_event_list[h]  # 一个窗口下的事件数组
        label_window = expanded_label_list[h]  # 一个窗口下的标签数组

        # 判断root在node_window中有无重复索引，有则合并到一个数组，没有则创建新的数组添加
        cc = node_window
        from collections import defaultdict
        dd = defaultdict(list)
        ee = defaultdict(list)
        for k, va in [(v, i) for i, v in enumerate(cc)]:
            dd[k].append(enent_window[va])
            ee[k].append(label_window[va])

        # print(dd)
        nodes_dict = list(dd.keys())
        event_dict = list(dd.values())
        label_dict = list(ee.values())

        for key in range(len(nodes_dict)):
            # 判断最大长度
            if len(event_dict[key]) <= 1:
                # print(event_dict[key])
                # print(label_dict[key])
                big_nodes.append(nodes_dict[key])
                big_events.append(event_dict[key])
                big_labels.append(label_dict[key])
                continue

            end_nodes.append(nodes_dict[key])
            end_events.append(event_dict[key])
            label = 0
            for lab in label_dict[key]:
                if lab:
                    label = 1
                    continue
            end_labels.append(label)

    #=============get labels and event count of each sliding window =========#

    # labels = []
    #
    # for j in range(inst_number):
    #     label = 0   #0 represent success, 1 represent failure
    #     for k in expanded_indexes_list[j]:
    #         # If one of the sequences is abnormal (1), the sequence is marked as abnormal
    #         if label_data[k]:
    #             label = 1
    #             continue
    #     labels.append(label)  # 一个时间窗口下的标签
    # assert inst_number == len(labels)
    # print("Among all instances, %d are anomalies"%sum(labels))
    print("删除：", + len(big_nodes))

    BGL_sequence = pd.DataFrame(columns=['node', 'sequence', 'label'])
    BGL_sequence['node'] = end_nodes
    BGL_sequence['sequence'] = end_events
    BGL_sequence['label'] = end_labels
    BGL_sequence.to_csv(para["BGL_sequence"], index=None)

if __name__ == "__main__":
    bgl_structured = load_BGL()
    bgl_sampling(bgl_structured)