import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

# Load the model pretrained on IMAGENET dataset.
resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

resnet50.eval().to(device)

# Prepare sample input data.
uris = [
    './hust/food/0a0ab85f-47f5-4e9c-b068-6e11e15c1fb6.jpg',
    './hust/food/0a0b2c3c-40c6-45a9-b253-8ba427414397.jpg',
    './hust/food/0a0c94da-7f5d-459c-ac48-3adba9995586.jpg',
    './hust/food/0a0fbcce-7fa8-4223-88d5-45e1449fdb18.jpg',
]

batch = torch.cat(
    [utils.prepare_input_from_uri(uri) for uri in uris]
).to(device)

# Run inference. Use pick_n_best(predictions=output, n=topN)
# helper function to pick N most probably hypothesis according to the model.
with torch.no_grad():
    output = torch.nn.functional.softmax(resnet50(batch), dim=1)

results = utils.pick_n_best(predictions=output, n=5)

# Display the result.
for uri, result in zip(uris, results):
    img = Image.open(os.path.join(uri))
    img.thumbnail((256,256), Image.ANTIALIAS)
    plt.imshow(img)
    plt.show()
    print(result)