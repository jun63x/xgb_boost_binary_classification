#coding: utf-8
from argparse import ArgumentParser
import numpy as np
import xgboost as xgb

# warningを出さない
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import sys

from collections import defaultdict
#import pprint



def make_id_to_feature_dict(X_train, Y_train, id_data_list):
    id_data_dict_list = []
    for index in id_data_list:
        count_dic = defaultdict(int)
        clicked_dic = defaultdict(int)
        for i, item in enumerate(X_train[:,index]):
            count_dic[item] += 1
            if Y_train[i] == 1:
                clicked_dic[item] += 1
        clicked_dic = {k:100*v/count_dic[k] for k, v in clicked_dic.items()}
        id_data_dict_list.append(clicked_dic)
    #pprint.pprint(id_data_dict_list[:2])
    return id_data_dict_list

def change_id_to_feature(X, id_data_dict_list, id_data_list):
    for index, dic in zip(id_data_list, id_data_dict_list):
        for item in X[:,index]:
            if item in dic:
                item = dic[item]
            else:
                item = 0
    return X

def load_data(path, size=None, test=False):
    with open(path, "rt") as f:        
        data = f.readlines()[1:]
        data = data[:size]
    X = []
    Y = []
    for datum in data:
        datum_list = datum.split(',')
        del datum_list[0]
        
        if datum_list[-4] == '':
            datum_list[-4] = sys.maxsize
        if datum_list[-3] == '':
            datum_list[-3] = sys.maxsize
        if test:
            del datum_list[-1]
        else:
            datum_list.append(datum_list.pop().replace('\n',''))
        datum_list = [int(item) for item in datum_list]
        
        if test:
            X.append(datum_list)
        else:
            X.append(datum_list[:13])
            Y.append(datum_list[-1])

    if test:
        return np.array(X)
    else:
        return np.array(X), np.array(Y)

        
def main():
    parser = ArgumentParser()
    parser.add_argument("--train", default="data_train.csv", type=str)
    parser.add_argument("--test", default="data_test.csv", type=str)
    parser.add_argument("--prediction", default="test_prediction.dat", type=str)
    parser.add_argument("--data_size", default=None, type=int)
    parser.add_argument('--change_id_to_feature', default=False, action='store_true')
    parser.add_argument('--ignore_unnecessary_features', default=False, action='store_true')
    args = parser.parse_args()

    X, Y = load_data(args.train, size=args.data_size)
    X, Y = shuffle(X, Y, random_state=0)
    X_test = load_data(args.test, test=True)

    if args.change_id_to_feature:
        id_data_list = [0, 1, 2, 3, 5, 7, 8, 12]
        dict_list = make_id_to_feature_dict(X, Y, id_data_list)
        X = change_id_to_feature(X, dict_list, id_data_list)            
        X_test = change_id_to_feature(X_test, dict_list, id_data_list)
        
    # scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    # dealete unnecessary features
    if args.ignore_unnecessary_features:
        ignore_feature_list = [1, 2]
        X = np.delete(X, ignore_feature_list, 1)
        X_test = np.delete(X_test, ignore_feature_list, 1)    
    
    # モデルのインスタンス化
    model = xgb.XGBClassifier(max_depth=7, n_estimators=1000)
    # trainデータを使ってモデルの学習
    model.fit(X, Y)
    
    # testデータの予測結果を出力
    test_prediction = np.array([item[1] for item in model.predict_proba(X_test)])
    with open(args.prediction, "w") as f:
        for item in test_prediction:
            f.write(str(item) + '\n')

        
if __name__ == "__main__":
    main()
