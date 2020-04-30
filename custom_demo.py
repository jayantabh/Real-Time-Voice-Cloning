from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm
import os
import pickle

encoder_weights = Path("encoder/saved_models/pretrained.pt")
vocoder_weights = Path("vocoder/saved_models/pretrained/pretrained.pt")
syn_dir = Path("synthesizer/saved_models/logs-pretrained/taco_pretrained")
encoder.load_model(encoder_weights)
synthesizer = Synthesizer(syn_dir)
vocoder.load_model(vocoder_weights)

print( " \n\n ##### Loaded Weights #####")

c_text = 0
c_audio = 0
npy_ctr = 1
all_ctr = 0

sampling_rate_list = []
audio_list = []
preprocessed_audio_list = []
all_text_list = []
embed_list = []
spectrogram_list = []

dataset_path = 'LibriSpeech/train-clean-100/'
for i in os.listdir(dataset_path):
    print( i, " : ", c_audio)
    if i=='103':
        print("Skipped")
        continue
    
    path_i = dataset_path + i
    for j in os.listdir(path_i): 
        
        #### SAVING ###############
        if all_ctr>=500:
            print( "\n ### Saving ###\n")
            np.save( "Numpy_data/fake_sampling_rate_"+str(npy_ctr)+".npy", np.array( sampling_rate_list ))
            np.save( "Numpy_data/fake_audio_"+str(npy_ctr)+".npy", np.array( audio_list ))
            np.save( "Numpy_data/fake_text_"+str(npy_ctr)+".npy", np.array( all_text_list ))
            np.save( "Numpy_data/fake_embed_"+str(npy_ctr)+".npy", np.array( embed_list ))
            np.save( "Numpy_data/fake_spectrogram_"+str(npy_ctr)+".npy", np.array( spectrogram_list ))
#             with open("Numpy_data/spectrogram_"+str(npy_ctr)+".txt", "wb") as fp:   #Pickling
#                 pickle.dump(spectrogram_list, fp)
#                 print(spectrogram_list[0] )
            np.save( "Numpy_data/preprocessed_audio_"+str(npy_ctr)+".npy", np.array( preprocessed_audio_list ))

            npy_ctr+=1
            all_ctr=0

            sampling_rate_list = []
            audio_list = []
            preprocessed_audio_list = []
            all_text_list = []
            embed_list = []
            spectrogram_list = []

        ##############################
            
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
        for k in tqdm(sorted_paths[:-1]):
            text = texts_list[text_idx]
            all_text_list.append(text)
            text_idx+=1
            in_fpath = path_j + "/" + k
            original_wav, sampling_rate = librosa.load(in_fpath)
            sampling_rate_list.append( sampling_rate )
            audio_list.append(original_wav )
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            preprocessed_audio_list.append(preprocessed_wav)
            embed = encoder.embed_utterance(preprocessed_wav)
            embed_list.append(embed)
            specs = synthesizer.synthesize_spectrograms([text], [embed])
            spectrogram_list.append(specs[0].T)
            #my_specs = np.array(specs)
            all_ctr+=1
            c_audio+=1
    
        if c_text!=c_audio:
            print( " Here :", path_j)

print(c_text, c_audio)
