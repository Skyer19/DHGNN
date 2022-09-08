import itertools
import re
from collections import Counter
import gensim
import numpy as np
import scipy.sparse as sp
import pickle
import os
from torch_geometric.data import Data
import jieba
import torch
jieba.set_dictionary('dict.txt.big')
from data import *
import dill as dill

w2v_dim = 300
max_len = 50

dic = {
    'non-rumor': 0,   # Non-rumor   NR
    'false': 1,   # false rumor    FR
    'unverified': 2,  # unverified tweet  UR
    'true': 3,    # debunk rumor  TR
}

def clean_str_cut(string, task='twitter'):

    if task != "weibo":
        string = re.sub(r"[^A-Za-z0-9(),!?#@\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)

    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    words = list(jieba.cut(string.strip().lower())) if task == "weibo" else string.strip().lower().split()
    return words


def read_replies(filepath, tweet_id, task, max_replies=30):
    filepath1 = filepath + "replies/" + tweet_id + ".txt"
    replies = []
    if os.path.exists(filepath1):
        with open(filepath1, 'r', encoding='utf-8') as fin:
            for line in fin:
                replies.append(clean_str_cut(line, task)[:max_len])  # 回复取50个字
    return replies[:max_replies]         # 取30个回复


def read_train_dev_test(root_path, file_name, appendix, X_all_tids):   #  'source tweet ID \t source tweet content \t label'
    filepath = root_path + file_name + appendix
    with open(filepath, 'r', encoding='utf-8') as fin:
        X_tid, X_content, X_replies, y_ = [], [], [], []
        for line in fin.readlines():
            tid, content, label = line.strip().split("\t")
            X_all_tids.append(tid)
            X_tid.append(tid)
            replies = read_replies(root_path, tid, file_name)
            X_replies.append(replies)
            X_content.append(clean_str_cut(content, file_name)[:max_len])   # 内容取50个字
            y_.append(dic[label])
    return X_tid, X_content, X_replies, y_

def construct_graph(root_path, file_name, X_all_tids, X_all_uids):
    graph = Graph()
    with open(root_path + file_name + "_graph.txt", 'r', encoding='utf-8') as input:
        edge_index, edges_weight = [], []
        for line in input.readlines():
            tmp = line.strip().split()
            src = tmp[0]  # 源推文id

            for dst_ids_ws in tmp[1:]:
                dst, w = dst_ids_ws.split(":")
                X_all_uids.append(dst)
                # edge_index.append([src, dst])
                edge_index.append([dst, src])
                # edges_weight.append(float(w))
                edges_weight.append(float(w))
    # X_tids = list(set(X_all_tids))
    X_uids = list(set(X_all_uids))
    X_id = list(X_all_tids + X_uids)
    num_node = len(X_id)
    print(num_node)
    # 微博用户总字典表
    X_id_dic = {id: i for i, id in enumerate(X_id)}

    # 构造节点类型 node_type 前1490为微博，后2968为用户
    t = list(set(X_all_uids))
    uid_temp = []
    for id in t:
        if len(id) >= 17:
            uid_temp.append(id)  # uid_temp 长度大于等于17的用户id
    for id in X_id:  # 0代表微博，1代表用户
        if len(id) >= 17 and id not in uid_temp:
            graph.node_type.append(0)
        else:
            graph.node_type.append(1)
    for _ in range(30):
        graph.node_type.append(1)
    # 构建边
    for tup in edge_index:
        graph.add_edge('weibo', 'user', X_id_dic[tup[1]], X_id_dic[tup[0]], relation_type='release')

    # 无向图
    edges_list = [[X_id_dic[tup[0]], X_id_dic[tup[1]]] for tup in edge_index]
    # edges_list_rev = [[X_id_dic[tup[1]], X_id_dic[tup[0]]] for tup in edge_index]
    # edges_list=list(edges_list_dir+edges_list_rev)

    # 构建边类型列表
    # graph.edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}  # type 序号
    # graph.edge_dict['self'] = len(graph.edge_dict)
    for _ in edge_index:
        # graph.edge_type += [graph.edge_dict[relation_type]]
        graph.edge_type.append(0)  # 目前只考虑发布

    edges_list = torch.LongTensor(edges_list).t()
    edges_weight = torch.FloatTensor(edges_weight)

    graph.node_type = torch.LongTensor(graph.node_type)
    graph.edge_type = torch.LongTensor(graph.edge_type)
    graph.edge_index = edges_list
    graph.edges_weight = edges_weight
    # graph.node_feature = torch.FloatTensor(graph.node_feature)
    # data = Data(edge_index=edges_list, edge_weight=edges_weight)
    return X_id_dic, graph
# 依照baseline构建
def test_construct_dygraph(dy_root_path, root_path, file_name, X_all_tids, X_all_uids):
    graph1 = Graph()
    graph2 = Graph()
    graph3 = Graph()
    edge_index1, edges_weight1 = [], []
    edge_index2, edges_weight2 = [], []
    edge_index3, edges_weight3 = [], []
    with open(root_path + file_name + "_graph.txt", 'r', encoding='utf-8') as input:
        for line in input.readlines():
            tmp = line.strip().split()
            src = tmp[0]  # 源推文id
            for twitter in os.listdir(dy_root_path):
                dy_src = twitter[:-4]  # 源推文id
                if dy_src==src:
                    with open(dy_root_path + twitter, 'r', encoding='utf-8') as input:  # 打开dy文件夹
                        for dst_ids_ws in tmp[1:]:  # 原始数据集用户
                            dst, w = dst_ids_ws.split(":")
                            index = 0        # 是否包含，标识
                            for line in input.readlines():
                                tmp = line.strip().split('->')
                                # 第一列
                                uid, tid, time = tmp[0].split(",")
                                uid = uid[2:-1]
                                uid2, tid2, time2 = tmp[1].split(",")
                                uid2 = uid2[2:-1]
                                time = int(float(time[2:-2]))
                                time2 = int(float(time2[2:-2]))
                                if (dst == uid or dst == uid2) and (time <= 30 or time2 <= 30):  # 如果存在且小于30分钟
                                    X_all_uids.append(dst)
                                    edge_index2.append([dst, src])
                                    index = 1
                                    break
                                elif (dst == uid or dst == uid2) and (time > 30 or time2 > 30): # 如果存在且大于30分钟
                                    X_all_uids.append(dst)
                                    edge_index3.append([dst, src])
                                    index = 1
                                    break
                            if index==0:
                                X_all_uids.append(dst)
                                edge_index1.append([dst, src])

    for edge in edge_index1:
        edge_index2.append(edge)
    for edge in edge_index2:
        edge_index3.append(edge)
    '''
    构造结点字典
    '''
    # X_tids = list(set(X_all_tids))
    X_uids = list(set(X_all_uids))
    X_id = list(X_all_tids + X_uids)
    num_node = len(X_id)
    print(num_node)
    # 微博用户总字典表
    X_id_dic = {id: i for i, id in enumerate(X_id)}

    '''
    构造节点类型 node_type 前1490为微博，后2968为用户
    '''
    # t = list(set(X_all_uids))
    # uid_temp = []
    # for id in t:
    #     if len(id) >= 17:
    #         uid_temp.append(id)  # uid_temp 长度大于等于17的用户id
    # for id in X_id:  # 0代表微博，1代表用户
    #     if len(id) >= 17 and id not in uid_temp:
    #         graph1.node_type.append(0)
    #     else:
    #         graph1.node_type.append(1)
    node_type = []
    for _ in X_all_tids:
        node_type.append(0)  # 0代表微博，1代表用户
    for _ in X_uids:
        node_type.append(1)
    graph1.node_type = node_type
    graph2.node_type = node_type
    graph3.node_type = node_type
    '''
    构建边
    '''
    for tup in edge_index1:
        graph1.add_edge('weibo', 'user', X_id_dic[tup[1]], X_id_dic[tup[0]], relation_type='release')
    for tup in edge_index2:
        graph2.add_edge('weibo', 'user', X_id_dic[tup[1]], X_id_dic[tup[0]], relation_type='release')
    for tup in edge_index3:
        graph3.add_edge('weibo', 'user', X_id_dic[tup[1]], X_id_dic[tup[0]], relation_type='release')

    # 无向图
    edges_list1 = [[X_id_dic[tup[0]], X_id_dic[tup[1]]] for tup in edge_index1]
    edges_list2 = [[X_id_dic[tup[0]], X_id_dic[tup[1]]] for tup in edge_index2]
    edges_list3 = [[X_id_dic[tup[0]], X_id_dic[tup[1]]] for tup in edge_index3]
    # edges_list_rev = [[X_id_dic[tup[1]], X_id_dic[tup[0]]] for tup in edge_index]
    # edges_list=list(edges_list_dir+edges_list_rev)

    '''
    构建边类型列表
    '''
    # graph.edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}  # type 序号
    # graph.edge_dict['self'] = len(graph.edge_dict)
    for _ in edge_index1:
        # graph.edge_type += [graph.edge_dict[relation_type]]
        graph1.edge_type.append(0)  # 目前只考虑发布
    for _ in edge_index2:
        # graph.edge_type += [graph.edge_dict[relation_type]]
        graph2.edge_type.append(0)  # 目前只考虑发布
    for _ in edge_index3:
        # graph.edge_type += [graph.edge_dict[relation_type]]
        graph3.edge_type.append(0)  # 目前只考虑发布

    edges_list = torch.LongTensor(edges_list1).t()
    edges_weight = torch.FloatTensor(edges_weight1)
    graph1.node_type = torch.LongTensor(graph1.node_type)
    graph1.edge_type = torch.LongTensor(graph1.edge_type)
    graph1.edge_index = edges_list
    graph1.edges_weight = edges_weight

    edges_list = torch.LongTensor(edges_list2).t()
    edges_weight = torch.FloatTensor(edges_weight2)
    graph2.node_type = torch.LongTensor(graph2.node_type)
    graph2.edge_type = torch.LongTensor(graph2.edge_type)
    graph2.edge_index = edges_list
    graph2.edges_weight = edges_weight

    edges_list = torch.LongTensor(edges_list3).t()
    edges_weight = torch.FloatTensor(edges_weight3)
    graph3.node_type = torch.LongTensor(graph3.node_type)
    graph3.edge_type = torch.LongTensor(graph3.edge_type)
    graph3.edge_index = edges_list
    graph3.edges_weight = edges_weight

    return X_id_dic, graph1, graph2, graph3

def test_liner_construct_dygraph(dy_root_path, root_path, file_name, X_all_tids, X_all_uids):
    graph1 = Graph()
    graph2 = Graph()
    graph3 = Graph()
    edge_index1, edges_weight1 = [], []
    edge_index2, edges_weight2 = [], []
    edge_index3, edges_weight3 = [], []
    tmp_clear = []
    for twitter in os.listdir(dy_root_path):
        src = twitter[:-4]  # 源推文id
        tmp_clear = []  # 去重复边
        num = -1        # 计数
        decay_rate = 2
        step = 1
        with open(dy_root_path + twitter, 'r', encoding='utf-8') as input:
            for line in input.readlines():
                num+=1
                # if len(tmp_clear) >= 40:
                #     break
                tmp = line.strip().split('->')
                # 第一列
                uid, tid, time = tmp[0].split(",")
                uid = uid[2:-1]
                time = float(time[2:-2])
                if time != 0.0 and num % step == 0 and uid not in tmp_clear:
                    tmp_clear.append(uid)
                    X_all_uids.append(uid)
                    edge_index1.append([uid, src])
                # 第二列
                uid2, tid2, time2 = tmp[1].split(",")
                uid2 = uid2[2:-1]
                time2 = float(time2[2:-2])  # 时延 分钟
                if time2 != 0.0 and num % step == 0 and uid2 not in tmp_clear:
                    step = step * decay_rate
                    X_all_uids.append(uid2)
                    edge_index1.append([uid2, src])
                    tmp_clear.append(uid2)
    with open(root_path + file_name + "_graph.txt", 'r', encoding='utf-8') as input:
        for line in input.readlines():
            tmp = line.strip().split()
            src = tmp[0]  # 源推文id
            for twitter in os.listdir(dy_root_path):
                dy_src = twitter[:-4]  # 源推文id
                if dy_src==src:
                    with open(dy_root_path + twitter, 'r', encoding='utf-8') as input:  # 打开dy文件夹
                        for dst_ids_ws in tmp[1:]:  # 原始数据集用户
                            dst, w = dst_ids_ws.split(":")
                            index = 0        # 是否包含，标识
                            for line in input.readlines():
                                tmp = line.strip().split('->')
                                # 第一列
                                uid, tid, time = tmp[0].split(",")
                                uid = uid[2:-1]
                                uid2, tid2, time2 = tmp[1].split(",")
                                uid2 = uid2[2:-1]
                                time = int(float(time[2:-2]))
                                time2 = int(float(time2[2:-2]))
                                if (dst == uid or dst == uid2) and (time <= 30 or time2 <= 30):
                                    X_all_uids.append(dst)
                                    edge_index2.append([dst, src])
                                    index = 1
                                    break
                                elif (dst == uid or dst == uid2) and (time > 30 or time2 > 30):
                                    X_all_uids.append(dst)
                                    edge_index3.append([dst, src])
                                    index = 1
                                    break
                            if index==0:
                                X_all_uids.append(dst)
                                edge_index1.append([dst, src])
                                tmp_clear.append(dst)



    for edge in edge_index1:
        edge_index2.append(edge)
    for edge in edge_index2:
        edge_index3.append(edge)
    '''
    构造结点字典
    '''
    # X_tids = list(set(X_all_tids))
    X_uids = list(set(X_all_uids))
    X_id = list(X_all_tids + X_uids)
    num_node = len(X_id)
    print(num_node)
    # 微博用户总字典表
    X_id_dic = {id: i for i, id in enumerate(X_id)}

    '''
    构造节点类型 node_type 前1490为微博，后2968为用户
    '''
    node_type = []
    for _ in X_all_tids:
        node_type.append(0)  # 0代表微博，1代表用户
    for _ in X_uids:
        node_type.append(1)
    graph1.node_type = node_type
    graph2.node_type = node_type
    graph3.node_type = node_type
    '''
    构建边
    '''
    for tup in edge_index1:
        graph1.add_edge('weibo', 'user', X_id_dic[tup[1]], X_id_dic[tup[0]], relation_type='release')
    for tup in edge_index2:
        graph2.add_edge('weibo', 'user', X_id_dic[tup[1]], X_id_dic[tup[0]], relation_type='release')
    for tup in edge_index3:
        graph3.add_edge('weibo', 'user', X_id_dic[tup[1]], X_id_dic[tup[0]], relation_type='release')

    # 无向图
    edges_list1 = [[X_id_dic[tup[0]], X_id_dic[tup[1]]] for tup in edge_index1]
    edges_list2 = [[X_id_dic[tup[0]], X_id_dic[tup[1]]] for tup in edge_index2]
    edges_list3 = [[X_id_dic[tup[0]], X_id_dic[tup[1]]] for tup in edge_index3]
    # edges_list_rev = [[X_id_dic[tup[1]], X_id_dic[tup[0]]] for tup in edge_index]
    # edges_list=list(edges_list_dir+edges_list_rev)

    '''
    构建边类型列表
    '''
    # graph.edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}  # type 序号
    # graph.edge_dict['self'] = len(graph.edge_dict)
    for _ in edge_index1:
        # graph.edge_type += [graph.edge_dict[relation_type]]
        graph1.edge_type.append(0)  # 目前只考虑发布
    for _ in edge_index2:
        # graph.edge_type += [graph.edge_dict[relation_type]]
        graph2.edge_type.append(0)  # 目前只考虑发布
    for _ in edge_index3:
        # graph.edge_type += [graph.edge_dict[relation_type]]
        graph3.edge_type.append(0)  # 目前只考虑发布

    edges_list = torch.LongTensor(edges_list1).t()
    edges_weight = torch.FloatTensor(edges_weight1)
    graph1.node_type = torch.LongTensor(graph1.node_type)
    graph1.edge_type = torch.LongTensor(graph1.edge_type)
    graph1.edge_index = edges_list
    graph1.edges_weight = edges_weight

    edges_list = torch.LongTensor(edges_list2).t()
    edges_weight = torch.FloatTensor(edges_weight2)
    graph2.node_type = torch.LongTensor(graph2.node_type)
    graph2.edge_type = torch.LongTensor(graph2.edge_type)
    graph2.edge_index = edges_list
    graph2.edges_weight = edges_weight

    edges_list = torch.LongTensor(edges_list3).t()
    edges_weight = torch.FloatTensor(edges_weight3)
    graph3.node_type = torch.LongTensor(graph3.node_type)
    graph3.edge_type = torch.LongTensor(graph3.edge_type)
    graph3.edge_index = edges_list
    graph3.edges_weight = edges_weight

    return X_id_dic, graph1, graph2, graph3

def construct_dygraph(dy_root_path, root_path, file_name, X_all_tids, X_all_uids):
    time_slice_1 = 30
    time_slice_2 = 120
    time_slice_3 = 300
    graph1 = Graph()
    edge_index1, edges_weight1 = [], []
    for twitter in os.listdir(dy_root_path):
        src = twitter[:-4]  # 源推文id
        tmp_clear = []  # 去重复边
        with open(dy_root_path + twitter, 'r', encoding='utf-8') as input:
            for line in input.readlines():
                if len(tmp_clear) >= 40:
                    break
                tmp = line.strip().split('->')
                # 第一列
                uid, tid, time = tmp[0].split(",")
                uid = uid[2:-1]
                time = int(float(time[2:-2]))
                if time != 0 and time <= time_slice_1 and uid not in tmp_clear:
                    tmp_clear.append(uid)
                    X_all_uids.append(uid)
                    edge_index1.append([uid, src])
                # 第二列
                uid2, tid2, time2 = tmp[1].split(",")
                uid2 = uid2[2:-1]
                time2 = int(float(time2[2:-2]))  # 时延 分钟
                if time2 <= time_slice_1 and uid2 not in tmp_clear:
                    tmp_clear.append(uid2)
                    X_all_uids.append(uid2)
                    edge_index1.append([uid2, src])

    graph2 = Graph()
    edge_index2, edges_weight2 = [], []
    for edge in edge_index1:
        edge_index2.append(edge)

    for twitter in os.listdir(dy_root_path):
        src = twitter[:-4]  # 源推文id
        tmp_clear = []  # 去重复边
        with open(dy_root_path + twitter, 'r', encoding='utf-8') as input:
            for line in input.readlines():
                if len(tmp_clear) >= 15:
                    break
                tmp = line.strip().split('->')
                # 第一列
                uid, tid, time = tmp[0].split(",")
                uid = uid[2:-1]
                time = int(float(time[2:-2]))
                if time != 0 and time > time_slice_1 and time <= time_slice_2 and uid not in tmp_clear:
                    tmp_clear.append(uid)
                    X_all_uids.append(uid)
                    edge_index2.append([uid, src])
                # 第二列
                uid2, tid2, time2 = tmp[1].split(",")
                uid2 = uid2[2:-1]
                time2 = int(float(time2[2:-2]))  # 时延 分钟
                if time2 > time_slice_1 and time2 <= time_slice_2 and uid2 not in tmp_clear:
                    tmp_clear.append(uid2)
                    X_all_uids.append(uid2)
                    edge_index2.append([uid2, src])

    graph3 = Graph()
    edge_index3, edges_weight3 = [], []
    for edge in edge_index2:
        edge_index3.append(edge)
    for twitter in os.listdir(dy_root_path):
        src = twitter[:-4]  # 源推文id
        tmp_clear = []  # 去重复边
        with open(dy_root_path + twitter, 'r', encoding='utf-8') as input:
            for line in input.readlines():
                if len(tmp_clear) >= 15:
                    break
                tmp = line.strip().split('->')
                # 第一列
                uid, tid, time = tmp[0].split(",")
                uid = uid[2:-1]
                time = int(float(time[2:-2]))
                if time != 0 and time > time_slice_2 and time <= time_slice_3 and uid not in tmp_clear:
                    tmp_clear.append(uid)
                    X_all_uids.append(uid)
                    edge_index3.append([uid, src])
                # 第二列
                uid2, tid2, time2 = tmp[1].split(",")
                uid2 = uid2[2:-1]
                time2 = int(float(time2[2:-2]))  # 时延 分钟
                if time2 > time_slice_2 and time2 <= time_slice_3 and uid2 not in tmp_clear:
                    tmp_clear.append(uid2)
                    X_all_uids.append(uid2)
                    edge_index3.append([uid2, src])
    '''
    构造结点字典
    '''
    # X_tids = list(set(X_all_tids))
    X_uids = list(set(X_all_uids))
    X_id = list(X_all_tids + X_uids)
    num_node = len(X_id)
    print(num_node)
    # 微博用户总字典表
    X_id_dic = {id: i for i, id in enumerate(X_id)}

    '''
    构造节点类型 node_type 前1490为微博，后2968为用户
    '''
    # t = list(set(X_all_uids))
    # uid_temp = []
    # for id in t:
    #     if len(id) >= 17:
    #         uid_temp.append(id)  # uid_temp 长度大于等于17的用户id
    # for id in X_id:  # 0代表微博，1代表用户
    #     if len(id) >= 17 and id not in uid_temp:
    #         graph1.node_type.append(0)
    #     else:
    #         graph1.node_type.append(1)
    node_type=[]
    for _ in X_all_tids:
        node_type.append(0) # 0代表微博，1代表用户
    for _ in X_uids:
        node_type.append(1)
    graph1.node_type=node_type
    graph2.node_type=node_type
    graph3.node_type=node_type
    '''
    构建边
    '''
    for tup in edge_index1:
        graph1.add_edge('weibo', 'user', X_id_dic[tup[1]], X_id_dic[tup[0]], relation_type='release')
    for tup in edge_index2:
        graph2.add_edge('weibo', 'user', X_id_dic[tup[1]], X_id_dic[tup[0]], relation_type='release')
    for tup in edge_index3:
        graph3.add_edge('weibo', 'user', X_id_dic[tup[1]], X_id_dic[tup[0]], relation_type='release')

    # 无向图
    edges_list1 = [[X_id_dic[tup[0]], X_id_dic[tup[1]]] for tup in edge_index1]
    edges_list2 = [[X_id_dic[tup[0]], X_id_dic[tup[1]]] for tup in edge_index2]
    edges_list3 = [[X_id_dic[tup[0]], X_id_dic[tup[1]]] for tup in edge_index3]
    # edges_list_rev = [[X_id_dic[tup[1]], X_id_dic[tup[0]]] for tup in edge_index]
    # edges_list=list(edges_list_dir+edges_list_rev)

    '''
    构建边类型列表
    '''
    # graph.edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}  # type 序号
    # graph.edge_dict['self'] = len(graph.edge_dict)
    for _ in edge_index1:
        # graph.edge_type += [graph.edge_dict[relation_type]]
        graph1.edge_type.append(0)  # 目前只考虑发布
    for _ in edge_index2:
        # graph.edge_type += [graph.edge_dict[relation_type]]
        graph2.edge_type.append(0)  # 目前只考虑发布
    for _ in edge_index3:
        # graph.edge_type += [graph.edge_dict[relation_type]]
        graph3.edge_type.append(0)  # 目前只考虑发布

    edges_list = torch.LongTensor(edges_list1).t()
    edges_weight = torch.FloatTensor(edges_weight1)
    graph1.node_type = torch.LongTensor(graph1.node_type)
    graph1.edge_type = torch.LongTensor(graph1.edge_type)
    graph1.edge_index = edges_list
    graph1.edges_weight = edges_weight

    edges_list = torch.LongTensor(edges_list2).t()
    edges_weight = torch.FloatTensor(edges_weight2)
    graph2.node_type = torch.LongTensor(graph2.node_type)
    graph2.edge_type = torch.LongTensor(graph2.edge_type)
    graph2.edge_index = edges_list
    graph2.edges_weight = edges_weight

    edges_list = torch.LongTensor(edges_list3).t()
    edges_weight = torch.FloatTensor(edges_weight3)
    graph3.node_type = torch.LongTensor(graph3.node_type)
    graph3.edge_type = torch.LongTensor(graph3.edge_type)
    graph3.edge_index = edges_list
    graph3.edges_weight = edges_weight

    return X_id_dic, graph1, graph2 , graph3



def read_dataset(dy_root_path, root_path, file_name):
    X_all_tids = []        #全源推文id
    X_all_uids = []        #全用户id

    X_train_tid, X_train_content, X_train_replies, y_train = read_train_dev_test(root_path, file_name, ".train", X_all_tids)
    X_dev_tid, X_dev_content, X_dev_replies, y_dev = read_train_dev_test(root_path, file_name, ".dev", X_all_tids)
    X_test_tid, X_test_content, X_test_replies, y_test = read_train_dev_test(root_path, file_name, ".test", X_all_tids)
    '''
    根据baseline构图(best)
    '''
    X_id_dic, graph1, graph2, graph3 = test_construct_dygraph(dy_root_path, root_path, file_name, X_all_tids, X_all_uids)
    '''
    根据线性构图
    '''
    # X_id_dic, graph1, graph2, graph3 = test_liner_construct_dygraph(dy_root_path, root_path, file_name, X_all_tids,
    #                                                           X_all_uids)
    '''
    时间片构图
    '''
    # X_id_dic, graph1, graph2, graph3 = construct_dygraph(dy_root_path, root_path, file_name, X_all_tids, X_all_uids)
    '''
    构建异构图
    '''
    # X_id_dic , graph = construct_graph(root_path, file_name, X_all_tids, X_all_uids)

    X_train_tid = np.array([X_id_dic[tid] for tid in X_train_tid])
    X_dev_tid = np.array([X_id_dic[tid] for tid in X_dev_tid])
    X_test_tid = np.array([X_id_dic[tid] for tid in X_test_tid])

    return X_train_tid, X_train_content, X_train_replies, y_train, \
           X_dev_tid, X_dev_content, X_dev_replies, y_dev, \
           X_test_tid, X_test_content, X_test_replies, y_test, graph1, graph2, graph3


def vocab_to_word2vec(fname, vocab):
    """
    Load word2vec from Mikolov
    """
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    count_missing = 0
    for word in vocab:
        if model.__contains__(word):
            word_vecs[word] = model[word]
        else:
            #add unknown words by generating random word vectors
            count_missing += 1
            word_vecs[word] = np.random.uniform(-0.25, 0.25, w2v_dim)

    print(str(len(word_vecs) - count_missing)+" words found in word2vec.")
    print(str(count_missing)+" words not found, generated by random.")
    return word_vecs


def build_vocab_word2vec(sentences, w2v_path='numberbatch-en.txt'):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1] >= 2]
    vocabulary_inv = vocabulary_inv[1:]  # don't need <PAD>
    # Mapping from word to index
    word_to_ix = {x: i+1 for i, x in enumerate(vocabulary_inv)}

    print("embedding_weights generation.......")
    word2vec = vocab_to_word2vec(w2v_path, word_to_ix)     #
    embedding_weights = build_word_embedding_weights(word2vec, vocabulary_inv)
    return word_to_ix, embedding_weights


def build_word_embedding_weights(word_vecs, vocabulary_inv):
    """
    Get the word embedding matrix, of size(vocabulary_size, word_vector_size)
    ith row is the embedding of ith word in vocabulary
    """
    vocab_size = len(vocabulary_inv)
    embedding_weights = np.zeros(shape=(vocab_size+1, w2v_dim), dtype='float32')
    #initialize the first row
    embedding_weights[0] = np.zeros(shape=(w2v_dim,) )

    for idx in range(1, vocab_size):
        embedding_weights[idx] = word_vecs[vocabulary_inv[idx]]
    print("Embedding matrix of size "+str(np.shape(embedding_weights)))
    return embedding_weights


def build_input_data(X, word_to_ix, is_replies=False, max_replies=30):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    if not is_replies:
        X = [[0]*(max_len - len(sentence)) + [word_to_ix[word] if word in word_to_ix else 0 for word in sentence] for sentence in X]
    else:
        X = [ [[0]*max_len]* (max_replies - len(replies))  + [ [0]*(max_len - len(doc)) + [word_to_ix[word] if word in word_to_ix else 0 for word in doc] for doc in replies] for replies in X]
    return X

def get_all_content(X_train_tid, X_dev_tid, X_test_tid, X_train_content, X_dev_content, X_test_content, graph):
    X_all_tid = list(list(X_train_tid) + list(X_dev_tid) + list(X_test_tid))
    X_all_content_pre = list(list(X_train_content) + list(X_dev_content) + list(X_test_content))
    X_content_dic = {}
    i = 0
    for id in X_all_tid:
        X_content_dic[id] = X_all_content_pre[i]
        i = i + 1
    for i in range(1490):  # twitter15: 1490  twitter:818 weibo:4664
        graph.X_all_content.append(X_content_dic[i])

def get_graph_all_content(X_train_tid, X_dev_tid, X_test_tid, X_train_content, X_dev_content, X_test_content, graph1, graph2, graph3):
    X_all_content_pre = list(list(X_train_content) + list(X_dev_content) + list(X_test_content))
    graph1.X_all_content = X_all_content_pre
    graph2.X_all_content = X_all_content_pre
    graph3.X_all_content = X_all_content_pre

def w2v_feature_extract(dy_root_path, root_path, filename, w2v_path):
    X_train_tid, X_train_content, X_train_replies, y_train, \
    X_dev_tid, X_dev_content, X_dev_replies, y_dev, \
    X_test_tid, X_test_content, X_test_replies, y_test, graph1, graph2,\
    graph3 = read_dataset(dy_root_path, root_path, filename)

    print("text word2vec generation.......")
    text_data = X_train_content + X_dev_content + X_test_content + list(itertools.chain(*X_train_replies)) + list(itertools.chain(*X_dev_replies)) + list(itertools.chain(*X_test_replies))
    vocabulary, word_embeddings = build_vocab_word2vec(text_data, w2v_path=w2v_path)
    pickle.dump(vocabulary, open(root_path + "/vocab.pkl", 'wb'))
    print("Vocabulary size: "+str(len(vocabulary)))

    print("build input data.......")
    X_train_content = build_input_data(X_train_content, vocabulary)
    X_dev_content = build_input_data(X_dev_content, vocabulary)
    X_test_content = build_input_data(X_test_content, vocabulary)

    X_train_replies = build_input_data(X_train_replies, vocabulary, True)
    X_dev_replies = build_input_data(X_dev_replies, vocabulary, True)
    X_test_replies = build_input_data(X_test_replies, vocabulary, True)

    '''
    获取所有按序号排列的源微博文本
    '''
    get_graph_all_content(X_train_tid, X_dev_tid, X_test_tid, X_train_content, X_dev_content, X_test_content, graph1, graph2, graph3)
    # get_all_content(X_train_tid, X_dev_tid, X_test_tid, X_train_content, X_dev_content, X_test_content,graph)


    dill.dump(graph1, open(root_path+"/graph1.pkl", 'wb'))
    dill.dump(graph2, open(root_path + "/graph2.pkl", 'wb'))
    dill.dump(graph3, open(root_path + "/graph3.pkl", 'wb'))
    # pickle.dump(graph, open(root_path + "/graph.pkl", 'wb'))
    pickle.dump([X_train_tid, X_train_content, X_train_replies, y_train, word_embeddings], open(root_path+"/train.pkl", 'wb') )
    pickle.dump([X_dev_tid, X_dev_content, X_dev_replies, y_dev], open(root_path+"/dev.pkl", 'wb') )
    pickle.dump([X_test_tid, X_test_content, X_test_replies, y_test], open(root_path+"/test.pkl", 'wb') )


if __name__ == "__main__":
    w2v_feature_extract('./dytwitter15/', './twitter15/', "twitter15", "twitter_w2v.bin")
    # w2v_feature_extract('./dytwitter16/', './twitter16/', "twitter16", "twitter_w2v.bin")
    # w2v_feature_extract('./weibo/', "weibo", "weibo_w2v.bin")



