import pandas as pd
import sys
import math
import itertools
import numpy as np
import pickle
from collections import Counter
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
import json
import datetime
from pathlib import Path
from qsr.Distance_neighborhood import calculate_neighborhood_distance
from Graph_Clustering import get_cluster


def get_objpair_type(df):
    """
    Get pure object types of each object pair by removing the ID
    :param df: data
    :return: before: -33722_ice_rect_fat_1*-33978_stone_rect_fat_1; after: ice_rect_fat_1*_stone_rect_fat_1
    """
    s = df.split("*")
    if any(map(str.isdigit, s[0].split("_")[-1])):
        obj1 = '_'.join(s[ 0 ].split("_")[ 1: -1])
    else:
        obj1 = '_'.join(s[ 0 ].split("_")[ 1:])
    if any(map(str.isdigit, s[0].split("_")[-1])):
        obj2 = '_'.join(s[ 1 ].split("_")[ 1: -1])
    else:
        obj2 = '_'.join(s[ 1 ].split("_")[ 1:])

    return obj1 + "*" + obj2


def convert_to_list(df):
    """
    convert the dataframe to list of list for clustering
    :param df: dataframe
    :return: data rows represented by list of list of list
    """
    row_list = [ ]
    for index, rows in df.iterrows():
        my_list = [
            [ rows.rcc_diff, rows.direct_diff, rows.dist_diff, rows.exist_diff, rows.qtc_diff, rows.objpair_type ] ]
        row_list.append(my_list)
    return row_list

def check_thres_M_merge(data, thres, M):
    count = 0
    percent = []
    for j in range(5):
        l = [item[j] for item in data]
        f = Counter(l).most_common(1)[0]
        if f[1] >= thres*len(data):
            count +=1
            percent.append(f[1])
    if count >= M:
        percent.sort(reverse=True)
        result = sum(percent[:M])/(M*len(data))
        return '%.2f' % result
    else:
        return False


def hierachical_clustering(data1, data2):
    """
    Apply hierachical clustering on raw/graph clusterred data
    :param data1: cluster1
    :param data2: cluster2
    :return: cluster1 and cluster2 and distance between them
    """
    if data1[0][-1] != data2[0][-1]:
        return 0
    else:
        sum_distance = check_thres_M_merge(data1+data2, 0.95, 5)
        # if sum_distance ==1:
        #     print(len(data1), 1111, len(data2))
        if sum_distance:
            return sum_distance
        else:
            return 0


def merge_duplicated_data(df):
    
    df_noduplicated = df.drop_duplicates(keep=False)
    duplicateRowsDF = df[df.duplicated(keep=False)]
    # duplicateRowsDF['size'] = duplicateRowsDF.groupby(["rcc_diff", "direct_diff", "dist_diff", "exist_diff", "qtc_diff", "objpair_type"]).size().to_frame('size')
    
    duplicateRowsDF['size'] = duplicateRowsDF.groupby(["rcc_diff", "direct_diff", "dist_diff", "exist_diff", "qtc_diff", "objpair_type"])['rcc_diff'].transform('size')
    duplicateRowsDF = duplicateRowsDF.drop_duplicates()
    duplicateRowsls = duplicateRowsDF.values.tolist()
    duplicated_clusters = []
    for i in duplicateRowsls:
        temp = []
        for j in range(i[-1]):
            temp.append(i[:-1])
        duplicated_clusters.append(temp)
    noduplicated_clusters = [[i] for i in df_noduplicated.values.tolist()]
    
    return duplicated_clusters + noduplicated_clusters
    

if __name__ == '__main__':
    write_path = "result_170_10_hard_5_bird_only/"
    Path(write_path).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("data/temp_train_170_10_merged_nosequ.csv")
    df[ "objpair_type" ] = df[ "objectid_pair" ].apply(lambda x: get_objpair_type(x))
    print(df.shape)
    df = df[df['objpair_type'].str.contains("bird")]
    df_unique = df.drop(columns=["objectid_pair", 'temporal_start', 'temporal_end'])
    print("Initial data size is {}".format(df_unique.shape))
    print("start initial clustering at: ", datetime.datetime.now())
    data = merge_duplicated_data(df_unique)
    # data = convert_to_list(df_unique)
    # data = get_cluster(data, 6, 0.25)
    print("After merge duplicates, the data size is {}".format(len(data)))
    print("finish initial clustering at: ", datetime.datetime.now())

    with open(write_path + 'initial_clusters.pkl', 'wb') as f:
        pickle.dump(data, f)
    # with open(write_path + 'initial_clusters.pkl', 'rb') as f:
    #     data = pickle.load(f)
    new_data = {}
    for d in range(len(data)):
        new_data[d] = data[d]
    total_num = len(new_data)
    print(total_num)
    X = []
    Y = []
    # Initially calculate the distance of all the initial clusters
    initial_dist_dict = {}
    start_time = time.time()

    for i in range(len(data) - 1):
        for j in range(i+1, len(data)):
            dist = hierachical_clustering(data[i], data[j])
            if dist != 0:
                initial_dist_dict[(i, j)] = dist
        if i % 10000 == 0:
            print("Finished {}th data, time is {}".format(i, datetime.datetime.now()))
            print(len(initial_dist_dict))
    # for k, v in intial_dist_dict.items():
    #     if v >0:
    #         print(k,v)
    with open(write_path + 'initial_dist_dict.pkl', 'wb') as f:
        pickle.dump(initial_dist_dict, f)



    # # Read data from file:
    # data = json.load( open( "file_name.json" ) )
    v = list(initial_dist_dict.values())
    if max(v) != 0:
        max_pair = list(initial_dist_dict.keys())[v.index(max(v))]
    else:
        sys.exit("No similar clusters found!")

    # record the two clusters to be merged

    # remove all the distance with at least one element is from the two removed clusters
    for k in initial_dist_dict.copy():
        if max_pair[0] in k or max_pair[1] in k:
            del initial_dist_dict[k]

    cluster_merge = new_data[max_pair[0]] + new_data[max_pair[1]]
    initial_max_index = max(new_data)
    print(initial_max_index)
    new_data[initial_max_index+1] = cluster_merge
    del new_data[max_pair[0]]
    del new_data[max_pair[1]]

    # data.append(cluster_merge)

    for d in range(total_num):
        max_index = max(new_data)
        for i in new_data.keys():
            if i != max_index:
                dist = hierachical_clustering(new_data[i], new_data[max_index])
                if dist != 0:
                    initial_dist_dict[(i, max_index)] = dist
        # print(len(initial_dist_dict))
        v = list(initial_dist_dict.values())

        if v:
            max_dist = max(v)
            max_pair = list(initial_dist_dict.keys())[v.index(max_dist)]

            # remove all the distance with at least one element is from the two removed clusters
            for k in initial_dist_dict.copy():
                if max_pair[0] in k or max_pair[1] in k:
                    del initial_dist_dict[k]
            cluster_merge = new_data[max_pair[0]] + new_data[max_pair[1]]
            new_data[max_index+1] = cluster_merge
            del new_data[max_pair[0]]
            del new_data[max_pair[1]]


            print("The size of original dataset is {}, now is {}, max distance is {}, dict size is {}".format(total_num, len(new_data), max_dist, len(initial_dist_dict)), "Used time: {}".format(time.time() - start_time))
            X.append(len(new_data))
            Y.append(max_dist)

        else:
            with open(write_path + "data_{}.txt".format(len(new_data)), "wb") as fp:   #Pickling
                pickle.dump(list(new_data.values()), fp)
            print("The size of original dataset is {}, now is {}".format(total_num, len(new_data)), "Used time: {}".format(time.time() - start_time))
            with open(write_path + "X.txt", "wb") as fp:   #Pickling
                pickle.dump(X, fp)
            with open(write_path + "Y.txt", "wb") as fp:   #Pickling
                pickle.dump(Y, fp)

            plt.plot(X, Y, 'ro-')
            plt.title('The shortest distance and clusters')
            plt.xlabel('number of clusters')
            plt.ylabel('Shortest distance')
            plt.axis([max(X),min(X),min(Y),max(Y)])
            plt.show()
            print("Finish clustering at: ", datetime.datetime.now())
            sys.exit("No similar clusters found!")








