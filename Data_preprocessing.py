import os
import math
import glob
import pandas as pd


def merge_csv(ae):
	if ae:
		data_path="/home/richie/Desktop/pddl/distrcc_thres_new_{}_split_3".format("ae")
	else:
		data_path="/home/richie/Desktop/pddl/geojson/distrcc_thres_new_{}split_test_3_merged_novel_l2t5_naive".format("")
	"""
    merge the csv file for all the game levels
    :param data_path: the path store the individual csv files of all the game levels
    :return: merged data file
    """
	os.chdir(data_path)
	extension = 'csv'
	all_filenames = [ i for i in glob.glob('*.{}'.format(extension)) ]
	### only train on some levels
	all_filenames = ["level_{}.csv".format(i) for i in [c for c in range(0,30)] if "level_{}.csv".format(i) in os.listdir(data_path)]
	combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
	# export to csv
	# combined_csv.to_csv("combined.csv", index=False, encoding='utf-8')
	return combined_csv


def get_diff_rcc(df):
	"""
    get RCC difference between before and after state
    :param df: dataframe
    :return: categorical rcc difference value
    """
	# if df[ "rcc_start" ] == df[ "rcc_end" ]:
	# 	return 0
	# else:
	# 	return df[ "rcc_start" ] + "*" + df[ "rcc_end" ]
	return df[ "rcc_start" ] + "*" + df[ "rcc_end" ]


def get_diff_direct(df):
	"""
    get direct difference between before and after state
    :param df: dataframe
    :return: categorical direction difference value
    """
	# if df["direct_start"] == df["direct_end"]:
	# 	return 0
	# else:
	# 	return df["direct_start"] + "*" + df["direct_end"]
	return df["direct_start"] + "*" + df["direct_end"]
	
def get_diff_dist(df):
	"""
    get distance difference between before and after state
    :param df: dataframe
    :return: categorical distance difference value
    """
	# if df["dist_start"] == df["dist_end"]:
	# 	return 0
	# else:
	# 	return str(df["dist_start"]) + "*" + str(df["dist_end"])
	return str(df["dist_start"]) + "*" + str(df["dist_end"])
	
	
def get_diff_cdr(df):
	"""
    get CDR difference between before and after state
    :param df: dataframe
    :return: categorical CDR difference value
    """
	if df[ "cdr_start" ] == df[ "cdr_end" ]:
		return 0
	else:
		b_s = df[ "cdr_start" ].split(":")
		a_s = df[ "cdr_end" ].split(":")
		s1 = list(set(b_s).difference(set(a_s)))
		s2 = list(set(a_s).difference(set(b_s)))
		return ':'.join(s1[ 0: ]) + "-" + ':'.join(s2[ 0: ])


def categorical_qaspa_diff(df, thres_qaspa_list):
	"""
    get QASpA difference between before and after state
    :param df: dataframe
    :return: categorical QASpA difference value
    """
	# qaspa_diff_dict = {0: [ -10000, -301 ], 1: [ -300, -201 ], 2: [ -200, -151 ], 3: [ -150, -101 ], 4: [ -100, -71 ],
	#                    5: [ -70, -41 ],
	#                    6: [ -40, -21 ], 7: [ -20, -11 ], 8: [ -10, -1 ], 9: [ 0, 0 ], 10: [ 1, 10 ], 11: [ 11, 20 ],
	#                    12: [ 21, 40 ], 13: [ 41, 70 ], 14: [ 71, 100 ], 15: [ 101, 150 ], 16: [ 151, 200 ],
	#                    17: [ 201, 300 ], 18: [ 301, 10000 ]}
	qaspa_diff_dict = {0: [-10000, thres_qaspa_list[0]], 1: [thres_qaspa_list[0]+1, 0], 2: [1, thres_qaspa_list[1]],
	                   3: [thres_qaspa_list[1]+1, 10000]}
	diff = round(float(df[ "qaspa_end" ]) - float(df[ "qaspa_start" ]))
	diff_start = round(float(df[ "qaspa_start" ]))
	diff_end = round(float(df[ "qaspa_end" ]))
	diff_a = ""
	for key, value in qaspa_diff_dict.items():
		if value[0] <= diff_start <= value[ 1 ]:
			diff_xa= str(key)
	for key, value in qaspa_diff_dict.items():
		if value[0] <= diff_end <= value[1]:
			diff_a = diff_a + "-" +str(key)
	return diff_a


def categorical_qaspx_diff(df, thres_qaspx_list):
	"""
    get QASpX difference between before and after state
    :param df: dataframe
    :return: categorical QASpX difference value
    """
	# qaspX_diff_dict = {0: [ -10000, -10 ], 1: [ -9, -7 ], 2: [ -6, -4 ], 3: [ -3, -1 ], 4: [ 0, 0 ], 5: [ 1, 3 ],
	#                    6: [ 4, 6 ], 7: [ 7, 9 ], 8: [ 10, 10000 ]}
	qaspX_diff_dict = {0: [-10000, thres_qaspx_list[0]], 1: [thres_qaspx_list[0]+1, 0], 2: [1, thres_qaspx_list[1]],
	                   3: [thres_qaspx_list[1]+1, 10000]}
	diff_start = round(eval(df[ "qasp_start" ])[ 0 ])
	diff_end = round(eval(df[ "qasp_end" ])[ 0 ])
	diff_x = ""
	for key, value in qaspX_diff_dict.items():
		if value[ 0 ] <= diff_start <= value[ 1 ]:
			diff_x = str(key)
	for key, value in qaspX_diff_dict.items():
		if value[ 0 ] <= diff_end <= value[ 1 ]:
			diff_x = diff_x + "-" +str(key)
	return diff_x


def categorical_qaspy_diff(df, thres_qaspy_list):
	"""
	get QASpY difference between before and after state
    :param df: dataframe
    :return: categorical QASpY difference value
    """
	# qaspY_diff_dict = {0: [ -10000, -10 ], 1: [ -9, -5 ], 2: [ -4, -3 ], 3: [ -2, -2 ], 4: [ -1, -1 ], 5: [ 0, 0 ],
	#                    6: [ 1, 1 ], 7: [ 2, 2 ], 8: [ 3, 4 ], 9: [ 5, 9 ], 10: [ 10, 10000 ]}
	qaspY_diff_dict = {0: [-10000, thres_qaspy_list[0]], 1: [thres_qaspy_list[0]+1, 0], 2: [1, thres_qaspy_list[1]],
	                   3: [thres_qaspy_list[1]+1, 10000]}
	diff_start = round(eval(df[ "qasp_start" ])[ 1 ])
	diff_end = round(eval(df[ "qasp_end" ])[ 1 ])
	diff_y = ""
	for key, value in qaspY_diff_dict.items():
		if value[ 0 ] <= diff_start <= value[ 1 ]:
			diff_y = str(key)
	for key, value in qaspY_diff_dict.items():
		if value[ 0 ] <= diff_end <= value[ 1 ]:
			diff_y = diff_y + "-" +str(key)
	return diff_y


def one_hot_data(df, selected_feature=["rcc", "direct", "dist", "qtc", "exist"]):
	"""
	One-hot encoding of the selected features
	:param df: dataframe
	:param selected_feature: the categorical features that need to be one-hot encoded
	:return: 0/1 value
	"""
	for i in selected_feature:
		df = pd.concat([ df, pd.get_dummies(df[ "%s_diff" % i ], prefix="%s_diff" % i) ], axis=1)
		df = df.drop("%s_diff" % i, axis=1)
		#print(i, df.shape)
	return df
	
	
def get_diff_qtc(df):
	qtc_start = df["qtc_start"]
	# qtc_start.append(df["qasp_start"])
	qtc_end = df["qtc_end"]
	# qtc_end.append(df["qasp_end"])
	# if str(qtc_start) == str(qtc_end):
	# 	return 0
	# else:
	# 	return str(qtc_start) + "*" + str(qtc_end)
	return str(qtc_start) + "*" + str(qtc_end)


def get_diff_exist(df):
	# if df["exist_start"] == df["exist_end"]:
	# 	return 0
	# else:
	# 	return df["exist_start"] + "*" + df["exist_end"]
	return df["exist_start"] + "*" + df["exist_end"]
		
	
def preprocess_data(ae):
	df = merge_csv(ae)
	df["rcc_diff"] = df.apply(lambda x: get_diff_rcc(x), axis=1)
	df[ "direct_diff" ] = df.apply(lambda x: get_diff_direct(x), axis=1)
	df["dist_diff"] = df.apply(lambda x: get_diff_dist(x), axis=1)
	# df[ "cdr_diff" ] = df.apply(lambda x: get_diff_cdr(x), axis=1)
	df["exist_diff"] = df.apply(lambda x: get_diff_exist(x), axis=1)
	# df[ "qaspa_diff" ] = df.apply(lambda x: categorical_qaspa_diff(x, thres_qaspa), axis=1)
	# df[ "qaspX_diff" ] = df.apply(lambda x: categorical_qaspx_diff(x, thres_qaspx), axis=1)
	# df[ "qaspY_diff" ] = df.apply(lambda x: categorical_qaspy_diff(x, thres_qaspy), axis=1)
	# df[ "qtc_start" ] = df["center_start"].apply(lambda x: get_qtc_feature(x))
	# df[ "qtc_end"] = df["center_end"].apply(lambda x: get_qtc_feature(x))
	df[ "qtc_diff"] = df.apply(lambda x: get_diff_qtc(x), axis=1)
	if ae:
		df = one_hot_data(df)
	# remove_feature = ['rcc_start', 'rcc_end', 'dist_start', 'dist_end', 'direct_start', 'direct_end', 'center_start',
	#                   'center_end', 'cdr_start', 'cdr_end', 'qasz_start', 'qasz_end', 'qasp_start', 'qasp_end',
	#                   'qaspa_start', 'qaspa_end', 'exist_start', 'exist_end']
	remove_feature = ['rcc_start', 'rcc_end', 'dist_start', 'dist_end', 'direct_start', 'direct_end', 'qtc_start', 'qtc_end', 'exist_start', 'exist_end']
	df = df.drop(remove_feature, axis = 1)
	# df.to_csv("/home/richie/Desktop/temp.csv", index=False)
	return df

df = preprocess_data(ae=False)
df.to_csv("/home/richie/Desktop/pddl/geojson/data/temp_test_3_merged_novel_l2t5_naive.csv", index=False)