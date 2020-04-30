import numpy as np
import librosa as lb
from tqdm import tqdm
from encoder import inference as encoder
# a = np.ones((1,40,541))
# b = np.ones((1,40,500))


# at = np.ones((541,40))
# bt = np.ones((500,40))

# c=np.array( [ at, bt ]) 
# print(c.shape)
# print(c[0].shape)
# print(c[1].shape)



## Mel-filterbank
mel_window_length = 25  # In milliseconds
mel_window_step = 10    # In milliseconds
mel_n_channels = 40


## Audio
sampling_rate = 16000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 160     # 1600 ms
# Number of spectrogram frames at inference
inference_n_frames = 80     #  800 ms


## Voice Activation Detection
# Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
# This sets the granularity of the VAD. Should not need to be changed.
vad_window_length = 30  # In milliseconds
# Number of frames to average together when performing the moving average smoothing.
# The larger this value, the larger the VAD variations must be to not get smoothed out. 
vad_moving_average_width = 8
# Maximum number of consecutive silent frames a segment can have.
vad_max_silence_length = 6


## Audio volume normalization
audio_norm_target_dBFS = -30
'''
npy_ctr = 0
for i in tqdm(range(1,54)):
    npy_ctr+=1
    frame_list = []
    a = np.load( "Numpy_data/audio_" + str(i) +".npy",allow_pickle=True)
    
    s = np.load( "Numpy_data/sampling_rate_" + str(i) +".npy",allow_pickle=True)
    for j in tqdm(range(a.shape[0])):
        preprocessed_wav = encoder.preprocess_wav(a[j], s[j])
        frame_list.append(preprocessed_wav)
    frame_np = np.array(frame_list)
    if frame_np.shape[0]!=a.shape[0]:
        print("\n\n\n#######Error_Here#######\n\n\n")
    np.save( "Numpy_data/preprocessed_audio_"+str(npy_ctr)+".npy", frame_np)
preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
'''
npy_ctr = 0
for i in tqdm(range(1,54)):
    npy_ctr+=1
    frame_list = []
    a = np.load( "Numpy_data/preprocessed_audio_" + str(i) +".npy",allow_pickle=True)
    for j in range(a.shape[0]):
        frames = lb.feature.melspectrogram(a[j],sampling_rate,n_fft=int(sampling_rate * mel_window_length / 1000),hop_length=int(sampling_rate * mel_window_step / 1000),n_mels=mel_n_channels)
        frames = frames.astype(np.float32).T
        frame_list.append(frames)
#         print("Frame : ",frames.shape)
        
    frame_np = np.array(frame_list)
    np.save( "Numpy_data/spectogram_"+str(npy_ctr)+".npy", frame_np)

'''
# mfccs = lb.feature.mfcc(y=a[0], sr=sampling_rate, n_mfcc=40,n_fft=int(sampling_rate * mel_window_length / 1000),hop_length=int(sampling_rate * mel_window_step / 1000)).T
print(mfccs[0])
'''