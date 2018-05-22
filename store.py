import argparse
import csv
import os
import pdb
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xlrd
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


def get_data(features, folders, mouse_types, pos_int, class_type): 
    data = []
    positions = []
    values = []

    for folder in folders:
        for mouse_type in mouse_types:
            xls = xlrd.open_workbook("../Data/"+folder+"/"+mouse_type+".xlsx", on_demand=True)
            sheets = xls.sheet_names()
            for idx,_ in enumerate(sheets):
                df = pd.read_excel("../Data/"+folder+"/"+mouse_type+".xlsx", sheet_name=idx, skiprows=[0])
                values.append(df["Value"][0])
                try:
                    df = df[features]
                except:
                    continue
                data.append(df)
                if class_type=="az_vs_wt":
                    positions.append(pos_int[mouse_type])
                elif class_type=="reg":
                    positions.append(pos_int[folder])

    num_examples_in_class = [0]*len(pos_int)
    for val,pos in zip(values,positions):
        num_examples_in_class[pos] += val
    min_ex = min(num_examples_in_class)
    # print(num_examples_in_class)

    #normalizing and taking each row as 1 example
    X,Y = [],[]
    for df,pos in zip(data,positions):
        region_puncta = []
        region_puncta.extend(df.values.tolist())
        region_puncta = np.array(region_puncta)
        region_puncta /= np.max(region_puncta,axis=0)
        X.extend(region_puncta.tolist())
        Y.extend([pos]*region_puncta.shape[0])

    # '''
    #taking equal samples from each class
    X_eq_samp = []
    Y_eq_samp = []
    l = list(range(len(Y)))
    np.random.shuffle(l)
    X = np.array(X)[l]
    Y = np.array(Y)[l]
    num_examples_in_class = [0]*len(pos_int)
    for x,y in zip(X,Y):
        if num_examples_in_class[y]<min_ex:
            X_eq_samp.append(x)
            Y_eq_samp.append(int(y))
            num_examples_in_class[y] += 1
    X = X_eq_samp
    Y = Y_eq_samp
    # '''

    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.01)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train = np.insert(X_train,X_train.shape[1],np.array(Y_train).astype(int),axis=1)
    X_test = np.insert(X_test,X_test.shape[1],np.array(Y_test).astype(int),axis=1)
    for file_name,arr in zip(["X_train","X_test"],[X_train,X_test]):
        with open(file_name+".csv", "w",newline="") as f:
            writer = csv.writer(f)
            writer.writerows(arr)
    pdb.set_trace()
    '''
    return X,Y

def ml_model(X,Y): #ml model
    model1 = svm.LinearSVC()
    # model2 = svm.SVC()
    # model3 = RandomForestClassifier()
    # model4 = tree.DecisionTreeClassifier()

    X_train, X_dev, Y_train, Y_dev = train_test_split(X,Y,test_size=0.1)
    feat = ['Area', 'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Intensity Max', 'Intensity Mean', 'Intensity Median', 'Intensity Min', 'Intensity StdDev', 'Intensity Sum', 'Number of Vertices', 'Sphericity', 'Volume']
    for model in [model1]:
        scores = cross_val_score(model,X,Y,cv=10,scoring="accuracy")
        print(np.mean(scores))
        model.fit(X_train,Y_train)
        # preds = model.predict(X_dev)
        # print(sum(preds==Y_dev)/len(Y_dev))
        # l = np.argsort(model.feature_importances_)[::-1]
        l = np.argsort(model.coef_)[::-1]
        imp_feat = [feat[i] for i in l]
        print(imp_feat)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Neurohack")
    parser.add_argument("--class-type", type=str, required=True, help="type of classification") #Alz vs Wild type OR between regions
    args = parser.parse_args()

    region_int = {'SLM Distal apical dendrites':0,
        'SP-SO Proximal apical dendrites':1,
        'SP-SO Proximal basal dendrites':2,
        'SR Medial apical dendrites':3}
    mouse_int = {'AD':0, 'WT':1}
    '''
    features = ['Area', 'Distance from Origin', 'Distance to Image Border XY',
                'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Intensity Max',
                'Intensity Mean', 'Intensity Median', 'Intensity Min',
                'Intensity StdDev', 'Intensity Sum', 'Number of Vertices', 'Position X',
                'Position Y', 'Position Z', 'Sphericity', 'Volume']
    '''
    # '''
    features = ['Area', 
                'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Intensity Max',
                'Intensity Mean', 'Intensity Median', 'Intensity Min',
                'Intensity StdDev', 'Intensity Sum', 'Number of Vertices', 'Sphericity', 'Volume']
    # '''
    if args.class_type=="az_vs_wt":
        folders = ['SLM Distal apical dendrites', 'SP-SO Proximal apical dendrites', 'SP-SO Proximal basal dendrites', 'SR Medial apical dendrites']
        X,Y = get_data(features, folders, ["AD","WT"], mouse_int, args.class_type)
        ml_model(X,Y)
        for folder in folders:
            X,Y = get_data(features, [folder], ["AD","WT"], mouse_int, args.class_type)
            mean_vertices = [0,0]
            for x,y in zip(X,Y):
                mean_vertices[y] += (x[5]/(len(Y)/2))
            print("Mean vert ",mean_vertices)
            ml_model(X,Y)
    elif args.class_type=="reg":
        # mouse_types = ["AD","WT"]
        mouse_types = ["AD"]
        X,Y = get_data(features, region_int.keys(), mouse_types, region_int, args.class_type)
    else:
        print("Wrong class type")
        sys.exit(0)