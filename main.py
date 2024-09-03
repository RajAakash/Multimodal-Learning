import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load the embeddings from the files
text_embeddings_np = np.load('text_embeddings.npy')
image_embedding_np = np.load('image_embedding.npy')
audio_embedding_np = np.load('audio_embedding.npy')

# Convert NumPy arrays back to PyTorch tensors
text_embeddings = torch.tensor(text_embeddings_np)
image_embedding = torch.tensor(image_embedding_np)
audio_embedding = torch.tensor(audio_embedding_np)

# Concatenate embeddings
combined_embedding = torch.cat((text_embeddings, image_embedding, audio_embedding), dim=1)

# Fully connected layer for final classification
fc = nn.Linear(combined_embedding.size(1), num_classes)
output = F.softmax(fc(combined_embedding), dim=1)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(fc.parameters(), lr=0.001)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = fc(combined_embedding)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()


test_predictions = fc(test_combined_embeddings)
accuracy = accuracy_score(test_labels, test_predictions.argmax(dim=1))
f1 = f1_score(test_labels, test_predictions.argmax(dim=1), average='weighted')
conf_matrix = confusion_matrix(test_labels, test_predictions.argmax(dim=1))

print(f'Accuracy: {accuracy}, F1 Score: {f1}')
