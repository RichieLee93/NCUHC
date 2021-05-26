import pandas as pd
import math
import itertools
import numpy as np
import pickle
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score


def get_objpair_type(df):
    """
    Get pure object types of each object pair by removing the ID
    :param df: data
    :return: before: -33722_ice_rect_fat_1*-33978_stone_rect_fat_1; after: ice_rect_fat*_stone_rect_fat
    """
    s = df.split("*")
    obj1 = '_'.join(s[0].split("_")[1: -1])
    obj2 = '_'.join(s[1].split("_")[1: -1])
    return obj1 + "*" + obj2


def drop_duplicates(df):
    """
    remove data with the same QSR changes and object type
    :param df: data
    :return: data without duplicates
    """
    column_select = []
    print("Initial data size is {}".format(df.shape))
    for i in df.columns:
        if i != "objectid_pair" and i != "temporal_start" and i != "temporal_end":
            column_select.append(i)
    df = df.drop_duplicates(subset=column_select, keep='last')
    print("After remove duplicates, the data size is {}".format(df.shape))
    return df


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


def one_hot_other(df, selected_feature=["objpair_type"]):
    """
    One-hot encoding of the selected features
    :param df: dataframe
    :param selected_feature: the categorical features that need to be one-hot encoded
    :return: 0/1 value
    """
    for i in selected_feature:
        df = pd.concat([ df, pd.get_dummies(df[i], prefix= i) ], axis=1)
        df = df.drop(i, axis=1)
        #print(i, df.shape)
    return df

def get_types(data):
    with open(data, "rb") as f:
        data = pickle.load(f)
    typ = []
    for i in data:
        typ.append(i[0][-1])
    type_dict = {x:typ.count(x) for x in typ}
    return type_dict

if __name__ == '__main__':
    M = 5
    T = 0.5
    dist_file = {('5', '0.5'): 'data_0.639_68.txt', ('4', '0.5'): 'data_0.62_56.txt', ('3', '0.5'): 'data_0.642_48.txt',
                 ('2', '0.5'): 'data_0.667_48.txt', ('5', '0.7'): 'data_0.803_141.txt', ('4', '0.7'): 'data_0.78_101.txt',
                 ('3', '0.7'): 'data_0.8_57.txt', ('2', '0.7'): 'data_0.712_53.txt', ('5', '0.9'): 'data_0.92_373.txt',
                 ('4', '0.9'): 'data_0.917_185.txt', ('3', '0.9'): 'data_0.935_99.txt', ('2', '0.9'): 'data_0.933_73.txt',
                 ('5', '0.99'): 'data_0.996_682.txt', ('4', '0.99'): 'data_0.996_358.txt', ('3', '0.99'): 'data_0.993_202.txt',
                 ('5', '0.98'): 'data_0.987_647.txt', ('4', '0.98'): 'data_0.985_327.txt', ('3', '0.98'): 'data_0.987_182.txt',
                 ('5', '0.97'): 'data_0.981_598.txt', ('4', '0.97'): 'data_0.983_303.txt', ('3', '0.97'): 'data_0.983_167.txt',
                 ('5', '0.95'): 'data_0.961_517.txt', ('4', '0.95'): 'data_0.955_254.txt', ('3', '0.95'): 'data_0.962_135.txt',
                 ('5', '0.92'): 'data_0.936_424.txt', ('4', '0.92'): 'data_0.94_208.txt', ('3', '0.92'): 'data_0.953_113.txt',
                 ('5', '0.87'): 'data_0.895_311.txt', ('4', '0.87'): 'data_0.906_165.txt', ('3', '0.87'): 'data_0.916_84.txt',
                 ('5', '0.85'): 'data_0.886_280.txt', ('4', '0.85'): 'data_0.9_153.txt', ('3', '0.85'): 'data_0.916_81.txt',
                 ('5', '0.82'): 'data_0.862_237.txt', ('4', '0.82'): 'data_0.881_138.txt', ('3', '0.82'): 'data_0.889_71.txt',
                 ('5', '0.8'): 'data_0.833_205.txt', ('4', '0.8'): 'data_0.859_129.txt', ('3', '0.8'): 'data_0.868_67.txt',
                 ('5', '0.75'): 'data_0.8_172.txt', ('4', '0.75'): 'data_0.837_117.txt', ('3', '0.75'): 'data_0.829_59.txt'}
    write_path = "/home/richie/Desktop/result/red_bird_only/result_kmeans/withdist/noprepro/10_times/{}/result_170_10_duplicates_hard_new_{}/subclusters/".format(T, M)
    Path(write_path).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("/home/richie/Desktop/pddl/data/red_bird_only/temp_170_10_merge.csv")
    df[ "objpair_type" ] = df[ "objectid_pair" ].apply(lambda x: get_objpair_type(x))
    df = df.drop(["objectid_pair", "temporal_start", "temporal_end"], axis = 1)
    # seperate data by object type
    num_cluster_type_dict = get_types('/home/richie/Desktop/result/red_bird_only/withdist/noprepro/10_times/{}/result_170_10_duplicates_hard_new_{}/{}'.format(T, M, dist_file[(str(M), str(T))]))
    all_obj_type = list(set(df["objpair_type"].tolist()))
    for i in range(len(all_obj_type)):
        df_sub = df[df["objpair_type"] == all_obj_type[i]]
        df3 = df_sub
        df_sub = one_hot_data(df_sub)
        data = one_hot_other(df_sub)
        n_cluster = num_cluster_type_dict[all_obj_type[i]]
        print("Total {} data for clustering, number of data left is {}, target number of cluster is {}".format(data.shape[0], len(all_obj_type)-i-1, n_cluster))
        df_tr_std = data.apply(lambda x: x if np.std(x) == 0 else stats.zscore(x))
        estimator = KMeans(n_clusters=n_cluster).fit(df_tr_std)
        labels = estimator.labels_
        labels = [str(i) + "_" + str(la) for la in labels]
        df3['clusters'] = labels
        # calinski_score = calinski_harabasz_score(df_tr_std, estimator.labels_)
        # print("The score of the clustering is {}".format(calinski_score))
        df3.to_csv(write_path + "result_170_{}.csv".format(i), index=False)

