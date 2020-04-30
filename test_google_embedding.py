from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import torch.optim as optim
from sklearn.metrics import accuracy_score as acc
from torch.nn.utils import clip_grad_norm_

embeds = np.load("Numpy_data/embed_1.npy", allow_pickle=True)
speakers = np.load("Numpy_data/speaker_1.npy", allow_pickle=True)
for i in tqdm(range(2,54)):
    embeds=np.concatenate((embeds,np.load("Numpy_data/embed_" + str(i) + ".npy", allow_pickle=True)), axis=0)
    speakers=np.concatenate((speakers,np.load("Numpy_data/speaker_" + str(i) + ".npy", allow_pickle=True)), axis=0)
    
X_train, X_val, Y_train, Y_val = train_test_split(embeds, speakers, test_size=0.1, shuffle=True)
print(Y_val)    

class CancerData(Dataset):
    def __init__(self, data, label):
        self.X_data = data
        self.Y_data = label

    def __len__(self):
        return self.X_data.shape[0]

    def transform_image(self, x):
        return TF.to_tensor(x).type(torch.FloatTensor)

    def __getitem__(self, idx):
        data_batch = self.X_data[idx]
        lable_batch = self.Y_data[idx]
        return data_batch, lable_batch
    
print("\n\n Creating DataLoader ... ")
Data_train = CancerData(X_train, Y_train)
Data_val = CancerData(X_val, Y_val)

###################################
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LR_DECAY = 0.99
WEIGHT_DECAY = 0.000
Epochs = 500
model_hidden_size =  512
model_embedding_size = 256
model_num_layers = 2
##################################

data_loader_train = DataLoader(Data_train, batch_size=BATCH_SIZE, shuffle=True,num_workers=10)
data_loader_val = DataLoader(Data_val, batch_size=BATCH_SIZE, shuffle=True,num_workers=10)

X_batch , Y_batch = next(iter(data_loader_val))
print("\n\n Batch : ",type(X_batch),type(Y_batch), X_batch.shape, Y_batch.shape)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Network defition
        self.fc = nn.Linear(256,248)
        
    def forward(self, x):
        x = self.fc(x)
        return x
    
        
model = Network().cuda()

def train(epoch):
    
    #Train
    
    train_acc_sum = 0
    total_loss = 0 
    model.train()
    pred_y = torch.empty(0).long()
    act_y = torch.empty(0).long()
    for batch_idx, batch in enumerate(tqdm(data_loader_train)):
        
        data, targets = batch[0], batch[1]
        data, targets = data.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output,targets)
        loss.backward()
        
#         clip_grad_norm_(model.parameters(), 3, norm_type=2)
        optimizer.step()
        total_loss+=loss.item()
        output = output.cpu().argmax(axis=1)
        
        pred_y = torch.cat((pred_y,output))
        act_y = torch.cat((act_y,targets.cpu())) 
        
    train_accuracy = acc(act_y, pred_y)
    op_train_loss = total_loss / (batch_idx + 1)
#     print("Epcch : ", epoch, " Loss : ", op_train_loss, " Accuracy : ", train_accuracy)

    #Validate
    acc_sum = 0
    total_loss = 0 
    model.eval()
    pred_y = torch.empty(0).long()
    act_y = torch.empty(0).long()
    
    for batch_idx, batch in enumerate(data_loader_val):
        
        data, targets = batch[0], batch[1]
        data, targets = data.cuda(), targets.cuda()
        
        with torch.no_grad():
            output = model(data)
        output = output.cpu().argmax(axis=1)
        pred_y = torch.cat((pred_y,output))
        act_y = torch.cat((act_y,targets.cpu())) 

    # op_val_loss = total_loss / (batch_idx + 1)

    test_accuracy = acc(act_y, pred_y)
    
    print("\nEpoch - ",epoch, " Train Accuracy : ", train_accuracy, " Val Accuracy : ", test_accuracy,"\n\n")

    return test_accuracy
    


optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = LR_DECAY)
criterion = F.cross_entropy

best_ep = 0
best_val_acc = -1

for i in range(Epochs+1):
    
    accuracy = train(i)
    
    scheduler.step()
    
    if accuracy>best_val_acc:
        best_val_acc = accuracy
        best_ep = i
        torch.save(model.state_dict(), "Kev_Model.pwf")

#     if i%10 == 0:
#         print("\n\n ----------- Best Values so far : Epoch - ",best_ep, "Val Accuracy : ", best_val_acc,"\n\n\n")


    
