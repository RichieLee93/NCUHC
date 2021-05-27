import numpy as np
import time
import os
import re
import json
import itertools
from pathlib import Path
from qsr.qmodels.Direction_Distance_center import directionDistanceRelations
from qsr.qmodels.RCC_new import compute_rcc_rel
from qsr.qmodels.CDR import CDRRelation
from qsr.qmodels.QASZ import QASZRelation
# from qsr.qmodels.QASP import QASPRelation
# from qsr.qmodels.QASP import QASPRelation
from qsr.qmodels.Speed import SpeedRelation
from qsr.qmodels.QASP_Angular import QASpARelation
from collections import defaultdict

def main(data, index):
    """Generate RCC relations and direction relations of objects in groundtruth json file
    :param data: Loaded json file of groudtruth.json
    :return: Dictionaries of RCC and direction of all objects
    """

    filtered_data = [{"code": i["properties"]["id"], "typeobj": i["properties"]["label"].split("*")[0], "velocity": i["properties"]["label"].split("*")[-1], "vertices": i["geometry"]["coordinates"][0]} for i in data[0]["features"] if i["properties"]["label"].split("*")[0] not in ["Ground", "Slingshot", "Trajectory"]]



#     cdr_dict = {}
#     qasz_dict = {}
#     qasp_dict = {}
#     qaspa_dict = {}
    rcc_dict = {}
    direction_dict = {}
    distance_dict = {}
    center_dict = {}
    temporal_dict = {}
    pair_data = list(itertools.combinations(filtered_data, 2))
    for subset in pair_data:
        i = subset[0]
        j = subset[1]
        
        obj_name = str(i["code"].split("_")[-1]) + "_" + str(i["typeobj"]) + "*" + str(j["code"].split("_")[-1]) + "_" + str(j["typeobj"])
        rcc_dict[obj_name] = compute_rcc_rel(i["vertices"], j["vertices"])


        direction_dict[obj_name], distance_dict[obj_name], center_dict[obj_name] = directionDistanceRelations(i["vertices"], j["vertices"])
        # cdr_dict[obj_name] = CDRRelation(i["vertices"], j["vertices"])
        # qasz_dict[obj_name]= QASZRelation(i["vertices"], j["vertices"])
        qasp_dict[obj_name] = SpeedRelation(i["velocity"], j["velocity"])
        # qaspa_dict[obj_name] = QASpARelation(i["angularVelocity"], j["angularVelocity"])
        temporal_dict[obj_name] = index
    # return rcc_dict, direction_dict, distance_dict, center_dict, cdr_dict, qasz_dict, qasp_dict, qaspa_dict, temporal_dict
    return rcc_dict, direction_dict, distance_dict, center_dict, temporal_dict


if __name__ == '__main__':
    start_time = time.time()
    # data_dir = 'sciencebirdsframework-release-alpha0.4.1/ScienceBirds/linux/gt/'
    all_data = os.listdir(data_dir)

    level_index_pair = [[re.split("\.|\_", i)[0], re.split("\.|\_", i)[1]] for i in all_data]

    levels_gt_dict = defaultdict(list)

    for i, j in level_index_pair:
        levels_gt_dict[i].append(int(j))
        levels_gt_dict[i] = sorted(levels_gt_dict[i])
    levels_gt_dict = dict(levels_gt_dict)
    for k, v in levels_gt_dict.items():
        if int(k) in [c for c in range(0, 5)]:
            for index in v:
                data_path = data_dir + "%s_%s_GTData.json"% (k, index)
                write_path = "/home/richie/Desktop/pddl/geojson/gt_batch_new_split_test_3_merged_naive/%s/"% k
                rcc_write_path = write_path+ "%s_rcc.json"%index
                direction_write_path = write_path+ "%s_direction.json"%index
                distance_write_path = write_path+ "%s_distance.json"%index
                center_write_path = write_path+ "%s_center.json"%index
                cdr_write_path = write_path+ "%s_cdr.json"%index
                qasz_write_path = write_path+ "%s_qasz.json"%index
                qasp_write_path = write_path+ "%s_qasp.json"%index
                qaspa_write_path = write_path + "%s_qaspa.json" % index
                temporal_write_path = write_path + "%s_temporal.json" % index
    
                Path(write_path).mkdir(parents=True, exist_ok=True)
                with open(data_path) as f:
                    data_gt = json.load(f)
                # rcc_relations, direction_relations, distance_relations, center_relations, cdr_relations, qasz_relations, qasp_relations, qaspa_relations, temporal_relations = main(data_gt, index)
                rcc_relations, direction_relations, distance_relations, center_relations, temporal_relations = main(data_gt, index)
                #
                with open(rcc_write_path, 'w') as fp:
                    fp.write(json.dumps(rcc_relations, sort_keys=True))
                with open(direction_write_path, 'w') as fp:
                    fp.write(json.dumps(direction_relations, sort_keys=True))
                with open(distance_write_path, 'w') as fp:
                    fp.write(json.dumps(distance_relations, sort_keys=True))
                with open(center_write_path, 'w') as fp:
                    fp.write(json.dumps(center_relations, sort_keys=True))
                # with open(cdr_write_path, 'w') as fp:
                #     fp.write(json.dumps(cdr_relations, sort_keys=True))
                # with open(qasz_write_path, 'w') as fp:
                #     fp.write(json.dumps(qasz_relations, sort_keys=True))
                # with open(qasp_write_path, 'w') as fp:
                #     fp.write(json.dumps(qasp_relations, sort_keys=True))
                # with open(qaspa_write_path, 'w') as fp:
                #     fp.write(json.dumps(qaspa_relations, sort_keys=True))
                with open(temporal_write_path, 'w') as fp:
                    fp.write(json.dumps(temporal_relations, sort_keys=True))
            #
            print("Finished level:", k)
    print("--- %s seconds ---" % (time.time() - start_time))


