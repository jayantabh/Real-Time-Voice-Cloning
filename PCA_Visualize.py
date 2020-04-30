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
from Trained_Models.model import Network
from sklearn.metrics import accuracy_score as acc
from sklearn.decomposition import PCA

my_encoder = Network().cuda()
my_encoder.load_state_dict(torch.load("Trained_Models/Kev_Model.pwf"))


print( "Visualize features")

i = 1
data_X = np.load("Numpy_data/spectogram_" + str(i) + ".npy", allow_pickle=True)
data_y = np.load("Numpy_data/speaker_" + str(i) + ".npy", allow_pickle=True)
data_xg = np.load("Numpy_data/embed_" + str(i) + ".npy", allow_pickle=True)
print(data_y)
# for i in tqdm([2]):
#     data_X=np.concatenate((data_X,np.load("Numpy_data/spectogram_" + str(i) + ".npy", allow_pickle=True)), axis=0)
#     data_y=np.concatenate((data_y,np.load("Numpy_data/speaker_" + str(i) + ".npy", allow_pickle=True)), axis=0)

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

data_xg = data_xg[:data_y.shape[0]]
print("\n\n Unique : ",np.unique(data_y).shape)
print("Shapes : ", data_X.shape, data_y.shape, data_xg.shape)

data_X =  TF.to_tensor(data_X).type(torch.FloatTensor).permute(1,2,0).unsqueeze(1)
print(data_X.shape)
with torch.no_grad():
    embeds =  my_encoder(data_X.cuda(),embed=True).cpu().numpy()
    
print(embeds.shape, data_y.shape, type(embeds), type(data_y))

pca = PCA(n_components=3)
pca_result = pca.fit_transform(embeds)
print(pca_result.shape, np.unique(data_y))

plt.scatter(pca_result[:,0], pca_result[:,1], c=data_y)
plt.savefig("Visualizations/pca.png")

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=pca_result[:,0],
    ys=pca_result[:,1],
    zs=pca_result[:,2], 
    c=data_y
)
plt.savefig("Visualizations/pca3d.png")

