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
from sklearn.model_selection import train_test_split
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
if torch.cuda.is_available():
    device = torch.device('cuda:1')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')
#%%
df = pd.read_csv("/home/maryam/drugmicro/micro_train_data12345.csv")
print(df.head)
#%%
smiles = df['SMILES']
#codIds = df['Drugs']
Lable = df['Label']
#%%
x_train, x_test, y_train, y_test = train_test_split(smiles, Lable, test_size=0.20, random_state=42)
#%%
#test_df = pd.DataFrame({'SMILES': x_train, 'Label': y_train})
#test_df.to_csv('/home/maryam/drugmicro/micro_train_data12345.csv', index=False)
#%%
#x_train,x_val, y_train,y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)
#%%
y_train = y_train.to_numpy()
#y_val = y_val.to_numpy()
y_test = y_test.to_numpy()
#%%
train_smile_graph = {}
train_label = []
train_smiles = []
for i,smile in enumerate(x_train):
    g = gd.smile_to_graph(smile)
    if g != None:
        train_smile_graph[smile] = g
        train_label.append(y_train[i])
        train_smiles.append(smile)
#%%
"""val_smile_graph = {}
val_label = []
val_smiles = [] 
for i,smile in enumerate(x_val):
    g = gd.smile_to_graph(smile)
    if g != None:
        val_smile_graph[smile] = g
        val_label.append(y_val[i])
        val_smiles.append(smile)"""
#%%
test_smile_graph = {}
test_label = []
test_smiles = []
for i,smile in enumerate(x_test):
    g = gd.smile_to_graph(smile)
    if g != None:
        test_smile_graph[smile] = g
        test_label.append(y_test[i])
        test_smiles.append(smile)   
#%%
train_data = MoleculeData(root='data', dataset='train_data_set',y=train_label,smile_graph=train_smile_graph,smiles=train_smiles)
#val_data = MoleculeData(root='data', dataset='val_data_set',y=val_label,smile_graph=val_smile_graph,smiles=val_smiles)
test_data = MoleculeData(root='data', dataset='test_data_set',y=test_label,smile_graph=test_smile_graph,smiles=test_smiles)
#%%
TRAIN_BATCH_SIZE = 32
train_loader   = DataLoader(train_data,batch_size=TRAIN_BATCH_SIZE,shuffle=True)
test_loader  = DataLoader(test_data,batch_size=TRAIN_BATCH_SIZE,shuffle=False)
#val_loader  = DataLoader(val_data,batch_size=TRAIN_BATCH_SIZE,shuffle=True)
#%%
#torch.save(test_data, "micro_data_graph_test_data.pt")
#%%
# training function at each epoch
def train(model, device, train_loader, optimizer, epoch,loss_fn):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    Loss = []
    for data in train_loader:
        data = data.to(device)
        y = data.y.view(-1, 1).long().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output = model(data)
        #pred = output.argmax(dim=-1)
        #print(output)
        #print(y)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        #scheduler.step()
        Loss.append(loss.item())
    nploss = np.asarray(Loss)
    avg_loss = np.average(nploss)
    return avg_loss
def predicting(model, device, loader,loss_fn):
    model.eval()
    total_loss=total_example=0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            y = data.y.view(-1, 1).long().to(device)
            y = y.squeeze(1)
            output = model(data)
            loss = loss_fn(output, y)
            total_loss+=loss
            total_example+=1
    return total_loss/total_example
#%%
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=2,num_features=114,n_filters=32, embed_dim=128, output_dim=64, dropout=0.4):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        D1_nn1 = Sequential(Linear(num_features, 198), ReLU(), Linear(198, 198))
        self.D1_conv1 = GINConv(D1_nn1)
        self.D1_bn1 = torch.nn.BatchNorm1d(198)

        D1_nn2 = Sequential(Linear(198, 64), ReLU(), Linear(64, 64))
        self.D1_conv2 = GINConv(D1_nn2)
        self.D1_bn2 = torch.nn.BatchNorm1d(64)

        D1_nn3 = Sequential(Linear(64, 32), ReLU(), Linear(32, 32))
        self.D1_conv3 = GINConv(D1_nn3)
        self.D1_bn3 = torch.nn.BatchNorm1d(32)


        # combined layers
        self.fc1 = nn.Linear(dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x1, edge_index_1, batch1 = data.x.float(), data.edge_index, data.batch
        x1 = F.relu(self.D1_conv1(x1, edge_index_1))
        x1 = self.D1_bn1(x1)
        x1 = self.dropout(x1)

        x1 = F.relu(self.D1_conv2(x1, edge_index_1))
        x1 = self.D1_bn2(x1)
        x1 = self.dropout(x1)

        x1 = F.relu(self.D1_conv3(x1, edge_index_1))
        x1 = self.D1_bn3(x1)
        x1 = global_add_pool(x1, batch1)
       
        # add some dense layers
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.fc2(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        out = self.out(x1)
        return out
#%%
model = GINConvNet().to(device)
print(model)
#%%
LR = 0.03
#eps_rate = 1.7687722582665366e-05
weight_decay = 1e-2
NUM_EPOCHS = 200
results = []
LOG_INTERVAL = 20
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=LR
                            ,
                    weight_decay=weight_decay)
#betas=(0.9,0.999),
best_ret = []
best_epoch = -1
model_st = "Gincovnet"
model_file_name = 'model_New' + model_st  +  '.model'
result_file_name = 'gcn_result1_' + model_st +  '.csv'
val_losses = []
train_losses = []
train_acc = []
val_acc = []
the_last_loss = 0
patience = 100
trigger_times = 0
count_loss_difference = 0

for epoch in range(NUM_EPOCHS):
    train_loss=train(model, device, train_loader, optimizer, epoch+1,loss_fn)
    test_loss = predicting(model, device, test_loader,loss_fn)
    train_T, train_S, train_Y = test_predicting(model, device, train_loader,loss_fn)
    val_T, val_S, val_Y = test_predicting(model, device, test_loader,loss_fn)
#     test_loss = predicting(model, device, test_loder,loss_fn)
#     print('Epoch% d: Train mae: %2.5f\t val mae: %2.5f\t test mae: %2.5f\t'
#           %(epoch, train_loss, val_loss.item(),test_loss.item()))
    train_ACC = accuracy_score(train_T, train_Y)
    val_ACC = accuracy_score(val_T, val_Y)
    print('Epoch% d: Train Loss: %2.5f\t Tarin_acc: %2.5f\t val Loss: %2.5f\t val_acc: %2.5f\t'
          %(epoch, train_loss,train_ACC,test_loss.item(),val_ACC))
    ret = [epoch,train_loss,test_loss.item()]
    
    train_losses.append(train_loss)
    val_losses.append(test_loss.item())
    train_acc.append(train_ACC)
    val_acc.append(val_ACC)
    # Early stopping
    the_current_loss = val_ACC
    
    if the_current_loss < the_last_loss:
        trigger_times += 1
        print('trigger times:', trigger_times)

        if trigger_times >= patience:
            print('Early stopping!\nStart to test process.')
            break
    else:
        
        trigger_times = 0
        the_last_loss = the_current_loss
        torch.save(model.state_dict(), 'micro-GINcovnet-2.model')
        

    with open(result_file_name,'w') as f:
        f.write(','.join(map(str,ret)))
#%%
model = GINConvNet().to(device)
path = "micro-GINcovnet-2.model"
model.load_state_dict(torch.load(path))
model.eval()
#%%
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_fn,}, path)
#%%
plt.figure(figsize=(10,5))
plt.title("GCNNet Training and Validation Loss")
plt.plot(val_losses,label="val")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.ylim(0.6,0.9)
plt.grid(True)
plt.savefig('p450-gnn-loss-new1.png', dpi=400,transparent=True)
plt.show()
#%%
plt.plot(train_acc,'-o')
plt.plot(val_acc,'-o')
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.legend(['Train','Valid'])
plt.ylim(0.65, 1)
plt.grid(True)
plt.savefig('accuracyGINConvNetTrainAndValidation-new1.png', dpi=400,transparent=True)
plt.title('Train vs Valid Accuracy')
plt.show()
#%%
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
test_T, test_S, test_Y = test_predicting(model, device, train_loader,loss_fn)
test_ACC = accuracy_score(test_T, test_Y)
print("P450-GNN-test" , test_ACC)
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
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#%%
from sklearn.metrics import recall_score
recall = recall_score(test_T, test_Y)
print(recall)
#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_T, test_Y)
print('P450-CNN\n\n', cm)
#%%
specificty =  cm[0,0]/(cm[0,0]+cm[0,1])
print("P450-CNN" , specificty)
#%%
from sklearn.metrics import f1_score
f_score = f1_score(test_T, test_Y, average=None)
print("F1 score for P450 CNN" , f_score)
#%%
print(test_T[1:50])
print(test_Y[1:50])
#%%
from sklearn.metrics import matthews_corrcoef
MCC = matthews_corrcoef(test_T, test_Y)
print("MCC of rfc p450 optuna test", MCC)
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
