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


print( "In model test")

data_X = np.load("Numpy_data/spectogram_1.npy", allow_pickle=True)
data_y = np.load("Numpy_data/speaker_1.npy", allow_pickle=True)
for i in tqdm(range(2,5)):
    data_X=np.concatenate((data_X,np.load("Numpy_data/spectogram_" + str(i) + ".npy", allow_pickle=True)), axis=0)
    data_y=np.concatenate((data_y,np.load("Numpy_data/speaker_" + str(i) + ".npy", allow_pickle=True)), axis=0)

l = []
for i in range(data_X.shape[0]):
    l.append( data_X[i].shape[0])

l=np.array(l)
idx = np.where(l>160)[0]

aa = data_X[idx]
new_aa = []
for i in range(aa.shape[0]):
    new_aa.append(aa[i][:160,:])
    
data_X = np.array(new_aa).astype(np.float32)    
data_y = data_y[idx]

print("\n\n Unique : ",np.unique(data_y).shape)

print( "\n\n Loaded Data ... ")
print( "Shapes : ", data_X.shape, data_y.shape)
print( "\n\n Splitting ... ")
# print(Y_val)    

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
        return self.transform_image(data_batch), lable_batch
    
print("\n\n Creating DataLoader ... ")
Data_val = CancerData(data_X, data_y)

###################################
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LR_DECAY = 0.95
WEIGHT_DECAY = 0.00
Epochs = 500
##################################

data_loader_train = DataLoader(Data_train, batch_size=BATCH_SIZE, shuffle=True,num_workers=10)
data_loader_val = DataLoader(Data_val, batch_size=BATCH_SIZE, shuffle=True,num_workers=10)

X_batch , Y_batch = next(iter(data_loader_val))
print("\n\n Batch : ",type(X_batch),type(Y_batch), X_batch.shape, Y_batch.shape)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Network defition
        self.conv_1 = nn.Conv2d(1, 32,3)
        self.bn_1 = nn.BatchNorm2d(16)
        self.mp_1 = nn.MaxPool2d(2)
        self.dp_1 = nn.Dropout(0.2)
        
        self.conv_2 = nn.Conv2d(32,64 ,3)
        self.bn_2 = nn.BatchNorm2d(32)
        self.mp_2 = nn.MaxPool2d(2)
        self.dp_2 = nn.Dropout(0.2)
        
        self.conv_3 = nn.Conv2d(64,128 ,2)
        self.bn_3 = nn.BatchNorm2d(64)
#         self.mp_3 = nn.MaxPool2d(2)
        self.dp_3 = nn.Dropout(0.2)
        
        self.conv_4 = nn.Conv2d(128,256, 2)
        self.bn_4 = nn.BatchNorm2d(128)
#         self.mp_4 = nn.MaxPool2d(2)
        self.dp_4 = nn.Dropout(0.2)
        
        self.conv_5 = nn.Conv2d(256,512, 2)
        self.bn_5 = nn.BatchNorm2d(256)
#         self.mp_5 = nn.MaxPool2d(2)
        self.dp_5 = nn.Dropout(0.2)
        
        self.conv_6 = nn.Conv2d(512,512, 2)
        
        self.conv_7 = nn.Conv2d(512,1024, 2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(1024,248)
        
    def forward(self, x):
        x = self.mp_1(F.relu(self.conv_1(x)))
        x = self.mp_2(F.relu(self.conv_2(x)))
        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        x = F.relu(self.conv_5(x))
        x = F.relu(self.conv_6(x))
        x = F.relu(self.conv_7(x))
        x = self.avg_pool(x)
        x = x.squeeze()
        x = self.fc(x)
        return x
        
    
        
model = Network().cuda()
model.load_state_dict(torch.load("Trained_Models/Trained_Model.pwf"))

def train(epoch):
    '''
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
    '''
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
    
    print("\nConv  Val Accuracy : ", test_accuracy,"\n\n")

    return test_accuracy
    


optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = LR_DECAY)
criterion = F.cross_entropy

best_ep = 0
best_val_acc = -1

for i in range(1):
    
    accuracy = train(i)
    
    scheduler.step()
    
    if accuracy>best_val_acc:
        best_val_acc = accuracy
        best_ep = i
        print("Saved")
        torch.save(model.state_dict(), "Kev_Model.pwf")

#     if i%10 == 0:
#         print("\n\n ----------- Best Values so far : Epoch - ",best_ep, "Val Accuracy : ", best_val_acc,"\n\n\n")


    
