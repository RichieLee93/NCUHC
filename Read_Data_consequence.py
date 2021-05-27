import os
import json
import pandas as pd
from datetime import datetime
import re
from collections import defaultdict
from pathlib import Path


def convert_dist(dist, thres_dist_list):

	dist_dict = {0: [0, thres_dist_list[0]], 1: [thres_dist_list[0]+1, thres_dist_list[1]], 2: [thres_dist_list[1]+1, 1000]}
	dist = round(float(dist))
	# print(111, dist_dict)
	dist_index = ""
	for k, v in dist_dict.items():
		if dist<=v[1] and dist>=v[0]:
			dist_index = k
	return dist_index

def read_data(thres_dist):
	path = "gt_batch_new_split_eval_gt_naive/"
	exist_path_ori = "gt_batch_new_exist_eval_gt_naive/"
	batch_level_dir = [int(lvl) for lvl in os.listdir(path)]
	levels_gt_dict = {}
	for i in batch_level_dir:
		npy_dir = path + str(i)
		gt_list = [int(j.split("_")[0]) for j in os.listdir(npy_dir)]
		gt_list = list(set(gt_list))
		gt_list.sort()
		levels_gt_dict[i] = gt_list
	remove_list = []
	use_list = set(batch_level_dir) - set(remove_list)
	# use_list = [i for i in range(60,90)]
	count = 0
	for level in use_list:
		count += 1
		print("Start level {}, current time is {}".format(level, datetime.now()))
		
		data_path = path + str(level) + "/"
		exist_path = exist_path_ori + str(level) + "/"
		all_data = os.listdir(data_path)
		# print(len(all_data))
		number_feature = 6
		total_num = int(len(all_data) / number_feature)
		# print(total_num)
		with open(data_path + "{}_distance.json".format(levels_gt_dict[ level ][ 0 ])) as rcc_start:
			data_0 = json.load(rcc_start)
		data_0 = {key: val for key, val in data_0.items() if val != "q"}
		rcc_all = [ ]
		dist_all = [ ]
		direct_all = [ ]
		center_all = []
# 		cdr_all = [ ]
# 		qasz_all = [ ]
# 		qasp_all = [ ]
# 		qaspa_all = [ ]
		exist_all = []
		temporal_all = []
		for t in range(total_num):
			with open(data_path + "{}_rcc.json".format(levels_gt_dict[ level ][ 0 ] + t)) as rcc:
				rcc_json = json.load(rcc)
				rcc_all.append(rcc_json)
			with open(data_path + "{}_distance.json".format(levels_gt_dict[ level ][ 0 ] + t)) as dist:
				dist_json = json.load(dist)
				dist_all.append(dist_json)
			with open(data_path + "{}_direction.json".format(levels_gt_dict[ level ][ 0 ] + t)) as direct:
				direct_json = json.load(direct)
				direct_all.append(direct_json)
			with open(data_path + "{}_qtc.json".format(levels_gt_dict[ level ][ 0 ] + t)) as center:
				center_json = json.load(center)
				center_all.append(center_json)
			# with open(data_path + "{}_cdr.json".format(levels_gt_dict[ level ][ 0 ] + t)) as cdr:
			# 	cdr_json = json.load(cdr)
			# 	cdr_all.append(cdr_json)
			# with open(data_path + "{}_qasz.json".format(levels_gt_dict[ level ][ 0 ] + t)) as qasz:
			# 	qasz_json = json.load(qasz)
			# 	qasz_all.append(qasz_json)
			# with open(data_path + "{}_qasp.json".format(levels_gt_dict[ level ][ 0 ] + t)) as qasp:
			# 	qasp_json = json.load(qasp)
			# 	qasp_all.append(qasp_json)
			# with open(data_path + "{}_qaspa.json".format(levels_gt_dict[ level ][ 0 ] + t)) as qaspa:
			# 	qaspa_json = json.load(qaspa)
			# 	qaspa_all.append(qaspa_json)
			with open(exist_path + "{}_exist.json".format(levels_gt_dict[ level ][ 0 ] + t)) as exist:
				exist_json = json.load(exist)
				exist_all.append(exist_json)
			with open(data_path + "{}_temporal.json".format(levels_gt_dict[ level ][ 0 ] + t)) as tempo:
				temporal_json = json.load(tempo)
				temporal_all.append(temporal_json)
		
		# print("Complete loading json files", count)
		state = []
		print(total_num)
		for key in data_0.keys():
		# 	start = 0
			for j in range(1, total_num):
				data_rcc_start = rcc_all[j-1]
				data_dist_start = dist_all[j-1]
				data_direct_start = direct_all[j-1]
				data_qtc_start = center_all[j-1]
				# data_cdr_start = cdr_all[j-1]
				# data_qasz_start = qasz_all[j-1]
				# data_qasp_start = qasp_all[j-1]
				# data_qaspa_start = qaspa_all[j-1]
				data_exist_start = exist_all[j-1]
				data_temporal_start = temporal_all[j-1]

				data_rcc_end = rcc_all[j]
				data_dist_end = dist_all[j]
				data_direct_end = direct_all[j]
				data_qtc_end = center_all[j]
				# data_cdr_end = cdr_all[j]
				# data_qasz_end = qasz_all[j]
				# data_qasp_end = qasp_all[j]
				# data_qaspa_end = qaspa_all[j]
				data_exist_end = exist_all[j]
				data_temporal_end = temporal_all[j]
				if key in data_rcc_end.keys() and key in data_rcc_start.keys():
					# if data_rcc_start[key] != data_rcc_end[key] or data_direct_start[ key ] != data_direct_end[ key ] or (str(data_exist_start[key.split("*")[0]])+"_"+ str(data_exist_start[key.split("*")[1]]) != str(data_exist_end[key.split("*")[0]])+"_"+ str(data_exist_end[key.split("*")[1]]) and  data_qtc_start[ key ] != data_qtc_end[ key ]) \
					# 		or convert_dist(data_dist_start[key], thres_dist) != convert_dist(data_dist_end[ key ], thres_dist):
					if data_rcc_start[key] != data_rcc_end[key] or (str(data_exist_start[key.split("*")[0]])+"_"+ str(data_exist_start[key.split("*")[1]]) != str(data_exist_end[key.split("*")[0]])+"_"+ str(data_exist_end[key.split("*")[1]]) and  data_direct_start[ key ] != data_direct_end[ key ]) \
							or convert_dist(data_dist_start[key], thres_dist) != convert_dist(data_dist_end[ key ], thres_dist):
						state.append(
							[ key, data_rcc_start[ key ], data_rcc_end[ key ], convert_dist(data_dist_start[ key ], thres_dist),
							  convert_dist(data_dist_end[ key ], thres_dist),
							  data_direct_start[ key ], data_direct_end[ key ], data_qtc_start[ key ], data_qtc_end[ key ],
							  # data_cdr_start[ key ],data_cdr_end[ key ],
							  # data_qasz_start[ key ], data_qasz_end[ key ],
							  # data_qasp_start[ key ],data_qasp_end[ key ],
							  # data_qaspa_start[ key ], data_qaspa_end[ key ],
							  str(data_exist_start[key.split("*")[0]])+"_"+ str(data_exist_start[key.split("*")[1]]), str(data_exist_end[key.split("*")[0]])+"_"+ str(data_exist_end[key.split("*")[1]]),
							  data_temporal_start[key], data_temporal_end[key]])


		if state:
			df = pd.DataFrame(state)

			df.columns = [ "objectid_pair", "rcc_start", "rcc_end", "dist_start", "dist_end", "direct_start", "direct_end",
					    "qtc_start", "qtc_end", "exist_start", "exist_end", "temporal_start", "temporal_end" ]
			write_path = "/home/richie/Desktop/pddl/geojson/distrcc_thres_new_split_eval_gt_level_2_type_2_604_naive/"
			Path(write_path).mkdir(parents=True, exist_ok=True)
			df.to_csv(write_path + "level_{}.csv".format(level), index=False)
			now = datetime.now()
			print("Finished level {}, current time is {}".format(level, now))



if __name__ == '__main__':
    
    #for thres_dist1 in [10]:
     #   for thres_dist2 in [80]:
    thres_dist_list = [10, 90]
    print(thres_dist_list)
    read_data(thres_dist_list)


