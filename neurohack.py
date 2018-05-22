import csv
import os
import pdb
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import xlrd
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable

def ml():
    data = []
    pos_int = {'SLM Distal apical dendrites':0,
    'SP-SO Proximal apical dendrites':1,
    'SP-SO Proximal basal dendrites':2,
    'SR Medial apical dendrites':3}
    positions = []
    values = []
    # model1 = svm.LinearSVC()
    # model2 = svm.SVC()
    model3 = RandomForestClassifier()
    # model4 = tree.DecisionTreeClassifier()

    if sys.argv[2]=="load":
        with open("save_data.pkl",'rb') as f:
            X = pickle.load(f)
        with open("save_pos.pkl",'rb') as f:
            Y = pickle.load(f)
        min_ex = 3506
    elif sys.argv[2]=="train":
        for folder in list(os.walk("../Data/"))[0][1]:
            xls = xlrd.open_workbook("../Data/"+folder+"/AD.xlsx", on_demand=True)
            sheets = xls.sheet_names()
            for idx,_ in enumerate(sheets):
                df = pd.read_excel("../Data/"+folder+"/AD.xlsx", sheet_name=idx, skiprows=[0])
                values.append(df["Value"][0])
                '''
                df = df[['Area', 'Distance from Origin', 'Distance to Image Border XY',
                    'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Intensity Max',
                    'Intensity Mean', 'Intensity Median', 'Intensity Min',
                    'Intensity StdDev', 'Intensity Sum', 'Number of Vertices', 'Position X',
                    'Position Y', 'Position Z', 'Sphericity', 'Volume']]
                '''
                # '''
                try:
                    df = df[['Area', 
                    'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Intensity Max',
                    'Intensity Mean', 'Intensity Median', 'Intensity Min',
                    'Intensity StdDev', 'Intensity Sum', 'Number of Vertices', 'Sphericity', 'Volume']]
                except:
                    continue
                    # pdb.set_trace()
                # '''
                data.append(df)
                positions.append(pos_int[folder])

        num_examples_in_class = [0,0,0,0]
        for val,pos in zip(values,positions):
            num_examples_in_class[pos] += val
        min_ex = min(num_examples_in_class)

        X,Y = [],[]
        for df,pos in zip(data,positions):
            region_puncta = []
            region_puncta.extend(df.values.tolist())
            region_puncta = np.array(region_puncta)
            region_puncta /= np.max(region_puncta,axis=0)
            X.extend(region_puncta.tolist())
            Y.extend([pos]*region_puncta.shape[0])

        with open("save_data.pkl",'wb') as f:
            pickle.dump(X,f)
        with open("save_pos.pkl",'wb') as f:
            pickle.dump(Y,f)
    else:
        print("what?")
        sys.exit(0)

    X_eq_samp = []
    Y_eq_samp = []
    l = list(range(len(Y)))
    np.random.shuffle(l)
    X = np.array(X)[l]
    Y = np.array(Y)[l]
    num_examples_in_class = [0,0,0,0]
    for x,y in zip(X,Y):
        if num_examples_in_class[y]<min_ex:
            X_eq_samp.append(x)
            Y_eq_samp.append(int(y))
            num_examples_in_class[y] += 1
    X = X_eq_samp
    Y = Y_eq_samp


    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1)
    feat = ['Area', 'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Intensity Max', 'Intensity Mean', 'Intensity Median', 'Intensity Min', 'Intensity StdDev', 'Intensity Sum', 'Number of Vertices', 'Sphericity', 'Volume']
    for model in [model3]:
        scores = cross_val_score(model,X,Y,cv=10,scoring="accuracy")
        print(np.mean(scores))
        model.fit(X_train,Y_train)
        l = np.argsort(model.feature_importances_)[::-1]
        imp_feat = [feat[i] for i in l]
        print(imp_feat)
    '''
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

class Feedforward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(12,9)
        self.lin2 = nn.Linear(9,6)
        self.lin3 = nn.Linear(6,2)
        self.relu = torch.nn.ReLU()
    def forward(self, data):
        out = data
        out = self.relu(self.lin1(out))
        out = self.relu(self.lin2(out))
        out = self.lin3(out)
        return out



class CustomDataset(Dataset):
    def __init__(self,data,labels):
        self.data = data
        self.labels = labels
    
    def __getitem__(self,index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]

def dl():
    data = []
    class_list = []
    values = []
    folder = "SR Medial apical dendrites"
    for class_idx,class_type in enumerate(["AD","WT"]):
        xls = xlrd.open_workbook("../Data/"+folder+"/"+class_type+".xlsx", on_demand=True)
        sheets = xls.sheet_names()
        for idx,_ in enumerate(sheets):
            df = pd.read_excel("../Data/"+folder+"/"+class_type+".xlsx", sheet_name=idx, skiprows=[0])
            values.append(df["Value"][0])
            try:
                '''
                df = df[['Area', 
                'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Intensity Max',
                'Intensity Mean', 'Intensity Median', 'Intensity Min',
                'Intensity StdDev', 'Intensity Sum', 'Number of Vertices', 'Sphericity', 'Volume']]
                '''
                df = df[['Intensity Mean','Number of Vertices']]
            except:
                continue
            data.append(df)
            class_list.append(class_idx)

    num_examples_in_class = [0,0]
    for val,pos in zip(values,class_list):
        num_examples_in_class[pos] += val
    min_ex = min(num_examples_in_class)

    X,Y = [],[]
    for df,pos in zip(data,class_list):
        region_puncta = []
        region_puncta.extend(df.values.tolist())
        region_puncta = np.array(region_puncta)
        region_puncta /= np.max(region_puncta,axis=0)
        X.extend(region_puncta.tolist())
        Y.extend([pos]*region_puncta.shape[0])

    X_eq_samp = []
    Y_eq_samp = []
    l = list(range(len(Y)))
    np.random.shuffle(l)
    X = np.array(X)[l]
    Y = np.array(Y)[l]
    num_examples_in_class = [0,0]
    for x,y in zip(X,Y):
        if num_examples_in_class[y]<min_ex:
            X_eq_samp.append(x)
            Y_eq_samp.append(int(y))
            num_examples_in_class[y] += 1
    X = X_eq_samp
    Y = Y_eq_samp
    
    '''
    # model1 = svm.LinearSVC()
    # model2 = svm.SVC()
    model3 = RandomForestClassifier()
    # model4 = tree.DecisionTreeClassifier()

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1)
    feat = ['Area', 'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Intensity Max', 'Intensity Mean', 'Intensity Median', 'Intensity Min', 'Intensity StdDev', 'Intensity Sum', 'Number of Vertices', 'Sphericity', 'Volume']
    for model in [model3]:
        scores = cross_val_score(model,X,Y,cv=10,scoring="accuracy")
        print(np.mean(scores))
        model.fit(X_train,Y_train)
        l = np.argsort(model.feature_importances_)[::-1]
        imp_feat = [feat[i] for i in l]
        print(imp_feat)
    '''

    # '''
    X_train, X_dev, Y_train, Y_dev = train_test_split(X,Y,test_size=0.25)
    X_dev, X_test, Y_dev, Y_test = train_test_split(X_dev,Y_dev,test_size=0.4)
    learning_rate = 0.01
    epochs = 21

    model = Feedforward()
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_dev = np.array(X_dev)
    Y_dev = np.array(Y_dev)
    dataset = CustomDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=16)
    dev_dataset = CustomDataset(X_dev, Y_dev)
    dev_dataloader = DataLoader(dev_dataset, batch_size=16)
    dataset = CustomDataset(X_test, Y_test)
    test_dataloader = DataLoader(dataset, batch_size=16)
    best_acc = 0
    for epoch in range(epochs):
        print("Train")
        acc = []
        tloss = []
        for batch in dataloader:
            data = Variable(torch.FloatTensor(np.array(batch[0])),requires_grad=True)
            optim.zero_grad()
            preds = model(data)
            labels = Variable(torch.LongTensor(batch[1]))
            loss = loss_fn(preds,labels)
            tloss.append(loss.data)
            loss.backward()
            optim.step()
            _,preds = torch.max(preds.data, 1)
            acc.append(sum(preds==labels.data)/len(preds))
        print(np.mean(acc),np.mean(tloss))
        print("Dev")
        acc = []
        dloss = []
        for batch in dev_dataloader:
            data = Variable(torch.FloatTensor(np.array(batch[0])))
            preds = model(data)
            labels = torch.LongTensor(batch[1])
            _,preds = torch.max(preds.data, 1)
            acc.append(sum(preds==labels)/len(preds))
        print(np.mean(acc))
        acc = np.mean(acc)
        if acc>best_acc:
            best_model = model
        
    print("Test")
    acc = []
    for batch in test_dataloader:
        data = Variable(torch.FloatTensor(np.array(batch[0])))
        preds = best_model(data)
        labels = torch.LongTensor(batch[1])
        _,preds = torch.max(preds.data, 1)
        acc.append(sum(preds==labels)/len(preds))
    print(np.mean(acc))
    # '''
if __name__=="__main__":
    if sys.argv[1]=="ml":
        ml()
    elif sys.argv[1]=="dl":
        dl()
    else:
        print("what?")
        sys.exit(0)