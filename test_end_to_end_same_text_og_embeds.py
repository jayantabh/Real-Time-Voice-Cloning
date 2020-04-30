from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm
import os
import pickle
import torch
from Trained_Models.model import Network
from sklearn.metrics import accuracy_score as acc

my_encoder = Network().cuda()
my_encoder.load_state_dict(torch.load("Trained_Models/Kev_Model.pwf"))

encoder_weights = Path("encoder/saved_models/pretrained.pt")
vocoder_weights = Path("vocoder/saved_models/pretrained/pretrained.pt")
syn_dir = Path("synthesizer/saved_models/logs-pretrained/taco_pretrained")
encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)


## Audio
mel_window_length = 25  # In milliseconds
mel_window_step = 10    # In milliseconds
mel_n_channels = 40
sampling_rate = 16000
partials_n_frames = 160     # 1600 ms
inference_n_frames = 80     #  800 ms
vad_window_length = 30  # In milliseconds
vad_moving_average_width = 8
vad_max_silence_length = 6

print( " \n\n ##### Loaded Weights #####")

c_text = 0
c_audio = 0
npy_ctr = 1
all_ctr = 0
speaker = -1
y_pred = []
y = []
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

for i in range(audio.shape[0]):
    preprocessed_wav = encoder.preprocess_wav(audio[i], sr[i])
    text = "This is a simulation of voice cloning and it does works."
    embed = encoder.embed_utterance(preprocessed_wav)
    specs = synthesizer.synthesize_spectrograms([text], [embed])
    generated_wav = vocoder.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    preprocessed_wav = encoder.preprocess_wav(generated_wav, synthesizer.sample_rate)
    frames = librosa.feature.melspectrogram(generated_wav,sampling_rate,n_fft=int(sampling_rate * mel_window_length / 1000),hop_length=int(sampling_rate * mel_window_step / 1000),n_mels=mel_n_channels)
    frames = frames.astype(np.float32).T[:160,:]
    t = torch.Tensor(frames).unsqueeze(0).unsqueeze(0)
    op = my_encoder(t.cuda())
    print("Gen ",i, ": ",op.shape, op.argmax(), speaker[i])
    y.append(op.argmax().item())
    y_pred.append(speaker[i])
            

print(acc(y,y_pred))
