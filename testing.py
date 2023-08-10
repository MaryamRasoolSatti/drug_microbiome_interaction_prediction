import pandas as pd 
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import networkx as nx
from torch_geometric.datasets import MoleculeNet
import matplotlib.pyplot as plt
import torch
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from tqdm.notebook import tqdm
import torch.nn as nn
import Data_Preprocessing.Graph_Data as gd 
from Data_Preprocessing.Graph_Data import MoleculeData
import seaborn as sns
from sklearn.metrics import accuracy_score
import torch.nn.functional as F 
from models.ginconv import GINConvNet
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
#%%
test_data = torch.load("/home/maryam/graph/micro_data_graph_test_data.pt")
#%%
test_loader  = DataLoader(test_data,batch_size=TRAIN_BATCH_SIZE,shuffle=False)

def test_predicting(model, device, loader_test,loss_fn):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader_test.dataset)))
    with torch.no_grad():
        for data in loader_test:
            data = data.to(device)
            output = model(data)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()

#%%
LR = 0.02
#eps_rate = 1.7687722582665366e-05
weight_decay = 1e-2
NUM_EPOCHS = 200
results = []
LOG_INTERVAL = 20
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=LR,weight_decay=weight_decay)

from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
model = GINConvNet().to(device)
path = "/home/maryam/graph/micro-GINcovnet-2.model"
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer= torch.optim.SGD(model.parameters(),lr=LR,
                    weight_decay=weight_decay)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
#%%
test_T, test_S, test_Y = test_predicting(model, device, test_loader,loss_fn)
test_ACC = accuracy_score(test_T, test_Y)
print(test_ACC)

#%%
from sklearn.metrics import classification_report
print(classification_report(test_T, test_Y))
#%%
from sklearn import metrics
accuracy = metrics.accuracy_score(test_T, test_Y)
print(accuracy)
#%%
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_T, test_Y)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc
from sklearn.metrics import recall_score
recall = recall_score(test_T, test_Y)
print(recall)
#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_T, test_Y)
print('CONF\n\n', cm)
#%%
specificty =  cm[0,0]/(cm[0,0]+cm[0,1])
print("specificity" , specificty)
#%%
from sklearn.metrics import f1_score
f_score = f1_score(test_T, test_Y, average=None)
print("F1 score " , f_score)
#%%
from sklearn.metrics import matthews_corrcoef
MCC = matthews_corrcoef(test_T, test_Y)
print("MCC", MCC)
#%%
from sklearn.metrics import balanced_accuracy_score

balanced_accuracy = balanced_accuracy_score(test_T, test_Y)
print("Balanced Accuracy:", balanced_accuracy)
#%%
from sklearn.metrics import precision_score

weighted_precision = precision_score(test_T, test_Y, average='weighted')
print("Weighted Precision:", weighted_precision)
#%%
from sklearn.metrics import recall_score

weighted_recall = recall_score(test_T, test_Y, average='weighted')
print("Weighted Recall:", weighted_recall)
