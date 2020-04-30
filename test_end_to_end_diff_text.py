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
dataset_path = 'LibriSpeech/train-clean-100/'
for i in tqdm(sorted(os.listdir(dataset_path))):
    speaker += 1
#     print( i, " : ", c_audio)
    if all_ctr==100:
        break
    if i=='103':
        speaker -= 1
        print("Skipped")
        continue
    
    print(i, speaker)
    path_i = dataset_path + i
    for j in os.listdir(path_i): 
        
        path_j = path_i + "/"+ j
        sorted_paths = sorted(os.listdir(path_j))
        
        ## Text Part ##
        texts_list = []
        text_file_path = path_j+ "/" + sorted_paths[-1]
        with open(text_file_path) as fp:
            line = fp.readline()
            while line:
                #print(line.split('-')[2][5:])
                texts_list.append(line)
                line = fp.readline()
                c_text += 1
        
        
        ## Audio Part ##
        text_idx = 0
        for num, k in enumerate(sorted_paths[:-1]):
            
            if num==5:
#                 print(num,k,text_idx)
                with torch.no_grad():
                    text = texts_list[text_idx]
                
                    in_fpath = path_j + "/" + k
                    original_wav, sampling_rate = librosa.load(in_fpath)
                    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
                    '''
                    ###
                    frames = librosa.feature.melspectrogram(preprocessed_wav,sampling_rate,n_fft=int(sampling_rate * mel_window_length / 1000),hop_length=int(sampling_rate * mel_window_step / 1000),n_mels=mel_n_channels)
                    frames = frames.astype(np.float32).T[:160,:]
                    t = torch.Tensor(frames).unsqueeze(0).unsqueeze(0)
                    embed = my_encoder(t.cuda(), embed=True)
#                     y.append(embed.argmax().item())
#                     y_pred.append(speaker)
                    embed_my = embed.cpu().numpy()
#                     print("Embed.shape:",embed.shape, torch.type(embed))
#                     print("Og : ",op.shape, op.argmax(), speaker)
                    ###
                    '''
                    embed = encoder.embed_utterance(preprocessed_wav)
#                     print("Type : ",(embed.max(),embed.min()),(embed_my.max(),embed_my.min()))
#                     specs = synthesizer.synthesize_spectrograms([text], [np.abs(embed_my)/100])
                    specs = synthesizer.synthesize_spectrograms([text], [embed])
                    generated_wav = vocoder.infer_waveform(specs[0])
                    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
                    preprocessed_wav = encoder.preprocess_wav(generated_wav, synthesizer.sample_rate)
                    frames = librosa.feature.melspectrogram(generated_wav,sampling_rate,n_fft=int(sampling_rate * mel_window_length / 1000),hop_length=int(sampling_rate * mel_window_step / 1000),n_mels=mel_n_channels)
                    frames = frames.astype(np.float32).T[:160,:]
                    t = torch.Tensor(frames).unsqueeze(0).unsqueeze(0)
                    op = my_encoder(t.cuda())
                    print("Gen : ",op.shape, op.argmax(), speaker)
                    y.append(op.argmax().item())
                    y_pred.append(speaker)
                    
                    all_ctr+=1
                    c_audio+=1
                    break
                    
            text_idx+=1
            
        break
            

print(c_text, c_audio)
print(y,y_pred)
print(acc(y,y_pred))
