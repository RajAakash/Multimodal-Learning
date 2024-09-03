import librosa
import torch
import torch.nn as nn
import numpy as np

# Load and process audio data
audio_data, sr = librosa.load(audio_path)
mfccs = librosa.feature.mfcc(audio_data, sr=sr)

# Define LSTM model
class AudioLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AudioLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h[:, -1, :])

audio_model = AudioLSTM(input_size=mfccs.shape[1], hidden_size=128, num_classes=num_classes)
audio_embedding = audio_model(torch.Tensor(mfccs).unsqueeze(0))
audio_embedding_np = audio_embedding.detach().cpu().numpy()
np.save('audio_embedding.npy', audio_embedding_np)