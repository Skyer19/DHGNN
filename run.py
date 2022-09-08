import os
import pickle
import torch
from sklearn.metrics import classification_report
from model.DHGNN import  DHGNN
from dataset.data import *


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def load_dataset(task):
    graph1 = renamed_load(open("dataset/" + task + "/graph1.pkl", 'rb'))
    graph2 = renamed_load(open("dataset/" + task + "/graph2.pkl", 'rb'))
    graph3 = renamed_load(open("dataset/" + task + "/graph3.pkl", 'rb'))

    X_train_tid, X_train_source, X_train_replies, y_train, word_embeddings = pickle.load(open("dataset/"+task+"/train.pkl", 'rb'))
    X_dev_tid, X_dev_source, X_dev_replies, y_dev = pickle.load(open("dataset/"+task+"/dev.pkl", 'rb'))
    X_test_tid, X_test_source, X_test_replies, y_test = pickle.load(open("dataset/"+task+"/test.pkl", 'rb'))
    config['embedding_weights'] = word_embeddings
    print("#nodes: ", len(graph1.node_type))
    return X_train_tid, X_train_source, X_train_replies, y_train, \
           X_dev_tid, X_dev_source, X_dev_replies, y_dev, \
           X_test_tid, X_test_source, X_test_replies, y_test, graph1, graph2, graph3


def train_and_test(model, task):
    model_suffix = model.__name__.lower().strip("text")
    config['save_path'] = 'checkpoint/weights.best.' + task + "." + model_suffix

    X_train_tid, X_train_source, X_train_replies, y_train, \
    X_dev_tid, X_dev_source, X_dev_replies, y_dev, \
    X_test_tid, X_test_source, X_test_replies, y_test, graph1, graph2, graph3 = load_dataset(task)

    nn = model(config, graph1, graph2, graph3)

    nn.fit(X_train_tid, X_train_source, X_train_replies, y_train,
           X_dev_tid, X_dev_source, X_dev_replies, y_dev)
    print("================================")
    nn.load_state_dict(torch.load(config['save_path']))
    y_pred = nn.predict(X_test_tid, X_test_source, X_test_replies)
    print(classification_report(y_test, y_pred, target_names=config['target_names'], digits=3))


config = {
    'lr':1e-3,
    'reg':0,
    'batch_size': 32,
    'dropout': 0.5,
    'maxlen':50,
    'epochs':30,
    'num_classes':4,
    'n_hid':400,
    'n_layers':4,  # HeteGT
    'n_heads':8,   # HeteGT注意头数
    'target_names':['NR', 'FR', 'UR', 'TR'],
    'num_layers': 1,   # GRU
    'n_out': 300,   # GRU output size
    'rnn': 'rum',    # gru or rum
    'n_node': 1490,
}


if __name__ == '__main__':
    task = 'twitter15'
    # task = 'twitter16'
    print("task: ", task)

    if task == 'twitter16':
        config['n_node'] = 818

    model = DHGNN
    train_and_test(model, task)
