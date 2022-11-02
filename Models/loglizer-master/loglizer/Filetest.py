import os

para = {
		'path': 'data/',  	  # data path
		'Path2': 'data2/BGL_2k.log_structured.csv',  	  # data path
		'log_file_name': 'BGL_2k.log',  # raw log data filename
		'log_event_mapping': 'BGL_2k.log_structured.csv',  # log event mapping relation list, obtained from log parsing
		'select_column': [0, 4],  # select the corresponding columns in the raw data
		'timeIndex': 1,  # the index of time in the selected columns, start from 0
		'timeInterval': 6,  # the size of time window with unit of hour
		'slidingWindow': 1,  # the size of sliding window interval with unit of hour
		'trainingSetPercent': 0.8,  # 80% of the time windows are used for training
		'tf-idf': False,  # whether turn on the tf-idf
		'balance': False,  # use balance mechanism during model building to solve imbalance problem
		'BGL_sequence': 'BGL_sequence.csv'
	}

print(para['BGL_sequence'])
# if not os.path.exists(para['save_path']):
# 	os.mkdir(para['save_path'])
# # log_size = raw_data.shape[0]
# # sliding_file_path = para['save_path'] + 'sliding_' + str(para['window_size']) + 'h_' + str(para['step_size']) + 'h.csv'