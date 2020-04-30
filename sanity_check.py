import numpy as np 
from tqdm import tqdm
import sys
import torch
import torchvision.transforms.functional as TF
import librosa
from sklearn.metrics import accuracy_score as acc
from Trained_Models.model import Network
from encoder import inference as encoder

mel_window_length = 25  # In milliseconds
mel_window_step = 10    # In milliseconds
mel_n_channels = 40
sampling_rate = 16000
partials_n_frames = 160     # 1600 ms
inference_n_frames = 80     #  800 ms
vad_window_length = 30  # In milliseconds
vad_moving_average_width = 8
vad_max_silence_length = 6

my_encoder = Network().cuda()
my_encoder.load_state_dict(torch.load("Trained_Models/Kev_Model.pwf"))


# a = np.ones((541,40)).astype(float)
# b = np.ones((500,40)).astype(float)
# c = np.array([a,b]) 
# print(c.dtype)
aa = np.load("Numpy_data/audio_1.npy", allow_pickle=True)
print(aa.dtype)

bb = np.load("Numpy_data/speaker_1.npy", allow_pickle=True)
cc = np.load("Numpy_data/sampling_rate_1.npy", allow_pickle=True)

for i in tqdm(range(2,20)):
    aa=np.concatenate((aa,np.load("Numpy_data/audio_" + str(i) + ".npy", allow_pickle=True)), axis=0)
    bb=np.concatenate((bb,np.load("Numpy_data/speaker_" + str(i) + ".npy", allow_pickle=True)), axis=0)
    cc=np.concatenate((cc,np.load("Numpy_data/sampling_rate_" + str(i) + ".npy", allow_pickle=True)), axis=0)
    
print(bb.shape,aa.shape)

idx = np.random.choice(range(1,aa.shape[0]), 100, replace=False)

audio = aa[idx]
speaker = bb[idx]
sr = cc[idx]
y_pred = []
y = []
for i in tqdm(range(audio.shape[0])):
    preprocessed_wav = encoder.preprocess_wav(audio[i], sr[i])                
    frames = librosa.feature.melspectrogram(preprocessed_wav,sampling_rate,n_fft=int(sampling_rate * mel_window_length / 1000),hop_length=int(sampling_rate * mel_window_step / 1000),n_mels=mel_n_channels)
    frames = frames.astype(np.float32).T[:160,:]
    t = torch.Tensor(frames).unsqueeze(0).unsqueeze(0)
    embed = my_encoder(t.cuda())
    y.append(embed.argmax().item())
    y_pred.append(speaker[i])
    
print(acc(y,y_pred))
