# Real-Time Voice Cloning 
## CS 7643 Deep Learning Project

### Parent Repositories:
WaveGlow: https://github.com/NVIDIA/waveglow.git
Real-Time-Voice-Cloning: https://github.com/CorentinJ/Real-Time-Voice-Cloning.git

### To Run:
Encoder: 
- Data Formatting: custom_create_data_for_encoder.py
- CNN: custom_conv_train_encoder.py
- LSTM: custom_LSTM_train_encoder.py

Vocoder:
Train: train_vocoder.ipynb Inference: inference_vocoder.ipynb (Google Colab)

### Files Modified from original repository:

./waveglow/inference.py
./waveglow/mel2samp.py
./waveglow/train.py
./waveglow/config.json
./waveglow/requirements.txt
./vocoder_train.py
./demo_cli.py

### Files Added to original repository:
./custom_conv_train_encoder.py
./custom_create_data_for_encoder.py
./custom_demo.py
./custom_LSTM_train_encoder.py
./custom_model_test.py
./PCA_Visualize.py
./PCA_Visualize_google.py
./sanity_check.py
./test_end_to_end_diff_text.py
./test_end_to_end_same_text_custom_embeddings.py
./test_end_to_end_same_text_og_embeds.py
./test_google_embedding.py
./test_mel_spectrogram.py


