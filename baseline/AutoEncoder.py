# from Read_Data_new import read_data
from Read_Data_consequence import read_data
from Data_preprocessing_autoencoder import preprocess_data
import Data_preprocessing
from sklearn.metrics import calinski_harabasz_score
from scipy import stats
import tensorflow as tf
import pickle
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import io_ops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli

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
        df = pd.concat([ df, pd.get_dummies(df[ "%s_start" % i ], prefix="%s_start" % i) ], axis=1)
        df = pd.concat([ df, pd.get_dummies(df[ "%s_end" % i ], prefix="%s_end" % i) ], axis=1)

        df = df.drop("%s_diff" % i, axis=1)
        df = df.drop("%s_start" % i, axis=1)
        df = df.drop("%s_end" % i, axis=1)

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



def sample_gumbel(shape, effect, eps=1e-20):
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  if effect:
    return -tf.log(-tf.log(U))/eps
  else:
    return (tf.log(U)-tf.log(1-U))/eps

def gumbel_softmax_sample(logits, temperature, effect):
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits), effect)
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard, effect):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature, effect)
  if hard:
    k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y


def model(input_dim_e, input_dim_s, k):
    K=k # number of classes
    N=1 # number of categorical distributions [100,200,1000]
    # input x (shape=(batch_size,input_dim))
    x = tf.placeholder(tf.float32,[None,input_dim_e])
    x2 = tf.placeholder(tf.float32,[None,input_dim_s])
    # x_n = tf.nn.batch_normalization(x, mean=0,variance=1, offset=None, scale=1, variance_epsilon=1e-16)
    # x_n = Bernoulli(logits=x_n)
    
    # variational posterior q(y|x), i.e. the encoder (shape=(batch_size,200))
    merg_x = tf.concat([x, x2], 1)
    net = slim.fully_connected(merg_x, 400, activation_fn=tf.nn.relu)
    net = tf.nn.batch_normalization(net, mean=0,variance=1, offset=None, scale=1, variance_epsilon=1e-16)
    net = tf.nn.dropout(net, 0.4)
    
    # net = slim.stack(x,slim.fully_connected,[512,256])
    
    # variational posterior q(y|x), i.e. the encoder (shape=(batch_size,200))
    merg_x = tf.concat([net, x2], axis=1)
    net = slim.fully_connected(merg_x, 400, activation_fn=tf.nn.relu)
    net = tf.nn.batch_normalization(net, mean=0,variance=1, offset=None, scale=1, variance_epsilon=1e-16)
    net = tf.nn.dropout(net, 0.4)
    # net = slim.stack(x,slim.fully_connected,[512,256])
    
    
    # merg_x = tf.concat([net, x2], axis=1)
    # net = slim.fully_connected(merg_x, 400, activation_fn=tf.nn.relu)
    # net = tf.nn.batch_normalization(net, mean=0,variance=1, offset=None, scale=1, variance_epsilon=1e-16)
    
    # unnormalized logits for N separate K-categorical distributions (shape=(batch_size*N,K))
    logits_y = tf.reshape(slim.fully_connected(net,K*N,activation_fn=None),[-1,K])
    # q_y = tf.nn.softmax(logits_y)
    # log_q_y = tf.log(q_y+1e-20)

    # temperature
    tau = tf.Variable(5.0,name="temperature")
    # sample and reshape back (shape=(batch_size,N,K))
    # set hard=True for ST Gumbel-Softmax
    y = tf.reshape(gumbel_softmax(logits_y,tau,hard=True, effect=True),[-1,N,K], name = "y")
    
    merg_x = tf.concat([slim.flatten(y), x2], 1)
    net = slim.fully_connected(merg_x, 400, activation_fn=tf.nn.relu)
    net = tf.nn.batch_normalization(net, mean=0,variance=1, offset=None, scale=1, variance_epsilon=1e-16)
    net = tf.nn.dropout(net, 0.4)
    
    # net = slim.stack(x,slim.fully_connected,[512,256])
    
    # variational posterior q(y|x), i.e. the encoder (shape=(batch_size,200))
    merg_x = tf.concat([net, x2], axis=1)
    net = slim.fully_connected(merg_x, 400, activation_fn=tf.nn.relu)
    net = tf.nn.batch_normalization(net, mean=0,variance=1, offset=None, scale=1, variance_epsilon=1e-16)
    # net = slim.stack(x,slim.fully_connected,[512,256])
    net = tf.nn.dropout(net, 0.4)
    
    
    
    # generative model p(x|y), i.e. the decoder (shape=(batch_size,200))
    # net = slim.stack(slim.flatten(y),slim.fully_connected,[256,512])
    
    # logits_x = slim.fully_connected(net,input_dim_e,activation_fn=None)
    # logits_x = tf.nn.batch_normalization(logits_x, mean=0,variance=1, offset=None, scale=1, variance_epsilon=1e-16)
    
    # (shape=(batch_size,8014))
    # p_x = Bernoulli(logits=logits_x)
    net = gumbel_softmax(net, tau, hard=True, effect=False)
    p_x = slim.fully_connected(net, input_dim_e, activation_fn=tf.nn.sigmoid)
    
    
    # loss and train ops
    # kl_tmp = tf.reshape(q_y*(log_q_y-tf.log(1.0/K)),[-1,N,K])
    # KL = tf.reduce_sum(kl_tmp,[1,2])
    # elbo=tf.reduce_sum(p_x.log_prob(x),1) - KL
    #
    # loss=tf.reduce_mean(-elbo)
    loss1 = tf.keras.losses.BinaryCrossentropy()
    loss = loss1(p_x,x)

    lr=tf.constant(0.001)
    train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,var_list=slim.get_model_variables())
    init_op=tf.global_variables_initializer()
    return init_op, train_op, loss, y, x, x2, tau, lr



########################Training###########################

if __name__ == '__main__':
 
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
    for mmm in [5, 4, 3]:
        for ttt in [0.87]:
            write_path = "/home/richie/Desktop/result/red_bird_only/result_autoencoder/withdist/noprepro/10_times/{}/result_170_10_duplicates_hard_new_{}/subclusters/".format(ttt, mmm)
            Path(write_path).mkdir(parents=True, exist_ok=True)
            df = pd.read_csv("/home/richie/Desktop/pddl/data/red_bird_only/temp_170_10_merge.csv")
            df[ "objpair_type" ] = df[ "objectid_pair" ].apply(lambda x: get_objpair_type(x))
            df = df.drop(["objectid_pair", "temporal_start", "temporal_end"], axis = 1)
            # seperate data by object type
            num_cluster_type_dict = get_types('/home/richie/Desktop/result/red_bird_only/withdist/noprepro/10_times/{}/result_170_10_duplicates_hard_new_{}/{}'.format(ttt, mmm, dist_file[(str(mmm), str(ttt))]))
            all_obj_type = list(set(df["objpair_type"].tolist()))
            for type in range(len(all_obj_type)):
                df_sub = df[df["objpair_type"] == all_obj_type[type]]
                df3 = df_sub
                ####map back diff to start and end###########
                df_sub[['rcc_start','rcc_end']] = df_sub["rcc_diff"].str.split("*",expand=True,)
                df_sub[['direct_start','direct_end']] = df_sub["direct_diff"].str.split("*",expand=True,)
                df_sub[['dist_start','dist_end']] = df_sub["dist_diff"].str.split("*",expand=True,)
                df_sub[['exist_start','exist_end']] = df_sub["exist_diff"].str.split("*",expand=True,)
                df_sub[['qtc_start','qtc_end']] = df_sub["qtc_diff"].str.split("*",expand=True,)
                df_sub = one_hot_data(df_sub)
                data = one_hot_other(df_sub)
        
                n_cluster = num_cluster_type_dict[all_obj_type[type]]
                if n_cluster <= 1:
                    df3["clusters"] = all_obj_type[type] + "_1"
                else:
                    print("number of clusters", n_cluster)
                    print("Total {} data for clustering, number of data left is {}, target number of cluster is {}".format(data.shape[0], len(all_obj_type)-type-1, n_cluster))
        
                    BATCH_SIZE=10
                    epoch =500
                    col_start = []
                    col_end = []
                    
                    for i in data.columns:
                        if "_start_" in i:
                            col_start.append(i)
                        if "_end_" in i:
                            col_end.append(i)
                    df_start = data[col_start]
                    frames = [df_start]*epoch
                    df_start = pd.concat(frames)
                    df_end = data[col_end]
                    frames = [df_end]*epoch
                    df_end = pd.concat(frames)
                    df_all = data[col_start+col_end]
                    test_array_s = df_start.values
                    test_array_e = df_end.values
                    data_size = data.shape[0]
            
                
                    NUM_ITERS= int(data_size*epoch/BATCH_SIZE +1)
                    tau0=1.0 # initial temperature
                    np_temp=tau0
                    np_lr=0.001
                    ANNEAL_RATE=0.00003
                    MIN_TEMP=0.5
                    k = n_cluster
                    dat=[]
                    init_op, train_op, loss, y, x, x2, tau, lr = model(test_array_e.shape[1], test_array_s.shape[1], k)
                    sess=tf.InteractiveSession()
                    sess.run(init_op)
                    label = []
                    for i in range(1,NUM_ITERS+1):
                        if i != NUM_ITERS-1:
                            start = (i-1)*BATCH_SIZE
                            end  = i*BATCH_SIZE
                        else:
                            start = (i-1)*BATCH_SIZE
                            end  = data_size*epoch
                        iter = int(data_size/BATCH_SIZE)
                        # slice = iter if i%iter !=0 else i%iter
                        # start = (slice -1)*100
                        # end  = slice*100
                        # print(start, end)
                        np_x=test_array_e[start:end,:]
                        np_x2=test_array_s[start:end,:]
                        l2_val = sess.run(y, {x: np_x, x2:np_x2})
                
                        _,np_loss=sess.run([train_op,loss],{
                          x:np_x,
                          x2:np_x2,
                          tau:np_temp,
                          lr:np_lr
                        })
                        # print(111,l2_val.shape, l2_val,3333, np.where(l2_val[0]==1),np.where(l2_val[0]==1)[1],np.where(l2_val[0]==1)[1][0] )
                        label += [np.where(l2_val[val]==1)[1][0] for val in range(l2_val.shape[0])]
                
                        # path = "/home/richie/Desktop/pddl/autoencoder_result/%d/"%k
                        # Path(path).mkdir(parents=True, exist_ok=True)
                    #     np.save(path+"%d.npy"%i,l2_val)
                
                        if i % 100 == 1:
                            dat.append([i,np_temp,np_loss])
                        if i % 1000 == 1:
                            np_temp=np.maximum(tau0*np.exp(-ANNEAL_RATE*i),MIN_TEMP)
                            np_lr*=0.9
                        if iter!=0 and i % iter == 1:
                            print('Step',i, 'ELBO:',  np_loss)
                    sess.close()
                    final_label = [all_obj_type[type] + "_" + str(label) for label in label[-data.shape[0]:]]
                    df3["clusters"] = final_label
                df3 = df3[["rcc_diff", "direct_diff", "dist_diff", "exist_diff", "qtc_diff", "objpair_type","clusters"]]
                df3.to_csv(write_path + "result_170_{}.csv".format(type), index=False)
            




