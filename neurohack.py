import csv
import os
import pdb
import pickle
import sys

import numpy as np
import pandas as pd
import xlrd
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

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

if sys.argv[1]=="load":
    with open("save_data.pkl",'rb') as f:
        X = pickle.load(f)
    with open("save_pos.pkl",'rb') as f:
        Y = pickle.load(f)
    min_ex = 3506
elif sys.argv[1]=="train":
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
            df = df[['Area', 
                'Ellipticity (oblate)', 'Ellipticity (prolate)', 'Intensity Max',
                'Intensity Mean', 'Intensity Median', 'Intensity Min',
                'Intensity StdDev', 'Intensity Sum', 'Number of Vertices', 'Sphericity', 'Volume']]
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
'''
for model in [model3, model4]:
    model.fit(X_train,Y_train)
    Y_pred = model.predict(X_test)
    print(sum(Y_pred==Y_test)/len(Y_test))
    pdb.set_trace()
'''
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
