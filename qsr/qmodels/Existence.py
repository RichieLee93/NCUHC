import numpy as np
import time
import os
import re
import json
from pathlib import Path

from collections import defaultdict


def main(data1, data2):
	"""Generate RCC relations and direction relations of objects in groundtruth json file
	:param data: Loaded json file of groudtruth.json
	:return: Dictionaries of RCC and direction of all objects
	"""
	# data = [eval(i) for i in data]

	exist_dict = {}

	
	for i in data1:
		obj_name = str(i)
		if i in data2:
			exist_dict[obj_name] = 1
		else:
			exist_dict[obj_name] = 0


	return exist_dict


if __name__ == '__main__':
	start_time = time.time()
	print("start at:" + str(start_time))
	data_dir = '/home/richie/Desktop/sciencebirdsframework-release-alpha0.4.1/ScienceBirds/linux/gt_level_2_type_2_604/'
	all_data = os.listdir(data_dir)
	
	level_index_pair = [ [ re.split("\.|\_", i)[ 0 ], re.split("\.|\_", i)[ 1 ] ] for i in all_data ]
	levels_gt_dict = defaultdict(list)
	
	for i, j in level_index_pair:
		levels_gt_dict[i].append(j)
		levels_gt_dict[i] = sorted(levels_gt_dict[ i ])
	levels_gt_dict = dict(levels_gt_dict)
	
	
	for k, v in levels_gt_dict.items():
		v = [int(j) for j in v]
		v.sort()
		json_all = []
		write_path = "/home/richie/Desktop/pddl/geojson/gt_batch_new_exist_eval_gt_level_2_type_2_604_naive/%s/" % k
		Path(write_path).mkdir(parents=True, exist_ok=True)
		for index in v:
			# print(index)
			data_path = data_dir + "%s_%d_GTData.json" % (k, index)
			with open(data_path) as f:
				data_gt = json.load(f)
			json_all.append([str(i["properties"]["id"].split("_")[-1]) + "_" + str(i["properties"]["label"].split("*")[0]) for i in data_gt[0]["features"] if "id" in i["properties"].keys() and i["properties"]["id"] != ''])
		for i in range(len(json_all)-1):
			exist_write_path = write_path + "%s_exist.json" % v[i]
			with open(exist_write_path, 'w') as fp:
				# json.dump(rcc_relations, fp)
				fp.write(json.dumps(main(json_all[i], json_all[i+1]), sort_keys=True))
		exist_write_path = write_path + "%s_exist.json" % v[-1]
		with open(exist_write_path, 'w') as fp:
			# json.dump(rcc_relations, fp)
			fp.write(json.dumps(main(json_all[-1], json_all[-1]), sort_keys=True))

		print("Finished level:", k, "time take:" + str(time.time() - start_time))
	print("--- %s seconds ---" % (time.time() - start_time))

