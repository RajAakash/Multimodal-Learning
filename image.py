import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load pre-trained ResNet-50 model
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)

# Process image
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image = preprocess(Image.open(image_path)).unsqueeze(0)
image_embedding = resnet(image)
image_embedding_np = image_embedding.detach().cpu().numpy()
np.save('image_embedding.npy', image_embedding_np)