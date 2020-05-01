# Real-Time Voice Cloning 
## CS 7643 Deep Learning Project

### Parent Repositories:
WaveGlow: https://github.com/NVIDIA/waveglow.git<br/>
Real-Time-Voice-Cloning: https://github.com/CorentinJ/Real-Time-Voice-Cloning.git<br/>

### To Run:
Encoder: 
- Data Formatting: custom_create_data_for_encoder.py
- CNN: custom_conv_train_encoder.py
- LSTM: custom_LSTM_train_encoder.py
- Full Pipeline with Custom Encoder: test_end_to_end_same_text_custom_embeddings.py

Vocoder:
- Train: train_vocoder.ipynb 
- Inference: inference_vocoder.ipynb (Google Colab)

### Files Modified from original repository:

./waveglow/inference.py<br/>
./waveglow/mel2samp.py<br/>
./waveglow/train.py<br/>
./waveglow/config.json<br/>
./waveglow/requirements.txt<br/>
./vocoder_train.py<br/>
./demo_cli.py<br/>

### Files Added to original repository:
./custom_conv_train_encoder.py<br/>
./custom_create_data_for_encoder.py<br/>
./custom_demo.py<br/>
./custom_LSTM_train_encoder.py<br/>
./custom_model_test.py<br/>
./PCA_Visualize.py<br/>
./PCA_Visualize_google.py<br/>
./sanity_check.py<br/>
./test_end_to_end_diff_text.py<br/>
./test_end_to_end_same_text_custom_embeddings.py<br/>
./test_end_to_end_same_text_og_embeds.py<br/>
./test_google_embedding.py<br/>
./test_mel_spectrogram.py


